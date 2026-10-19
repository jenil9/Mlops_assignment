"""Integration tests for FastAPI endpoints, authentication, and recommendations."""

from __future__ import annotations

import tempfile
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.state import LoadedModel
from src.common.config import Settings
from src.common.db import Feedback, get_engine, get_session
from src.recsys.bundle import build_bundle


@pytest.fixture
def fake_bundle():
    user_factors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    item_factors = np.array(
        [[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [0.1, 0.9], [0.5, 0.5]], dtype=np.float64
    )
    user_ids = np.array([1, 2], dtype=np.int64)
    item_ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)

    ratings_rows = []
    for mid in [10, 20, 30, 40, 50]:
        for uid in range(1, 15):
            ratings_rows.append(
                {"userId": uid, "movieId": mid, "rating": 4.0, "timestamp": 1000 + uid}
            )
    ratings_df = pd.DataFrame(ratings_rows)

    movies_df = pd.DataFrame(
        [
            {"movieId": 10, "title": "Movie 10", "genres": ["Action"]},
            {"movieId": 20, "title": "Movie 20", "genres": ["Action"]},
            {"movieId": 30, "title": "Movie 30", "genres": ["Comedy"]},
            {"movieId": 40, "title": "Movie 40", "genres": ["Comedy"]},
            {"movieId": 50, "title": "Movie 50", "genres": ["Drama"]},
        ]
    )

    params = {
        "model": {"factors": 2, "reg": 0.1, "min_item_ratings": 5},
        "serving": {
            "min_ratings_personalize": 3,
            "liked_threshold": 4.0,
            "because_rows": 3,
            "default_n": 10,
            "max_n": 50,
        },
    }

    model = {
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "global_mean": 3.5,
    }

    bundle = build_bundle(model, ratings_df, movies_df, params)
    return LoadedModel(bundle=bundle, version=1)


@pytest.fixture
def client(fake_bundle) -> Generator[TestClient, None, None]:
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_url = f"sqlite:///{tmp.name}"

        def fake_loader(settings: Settings) -> LoadedModel:
            return fake_bundle

        app = create_app(database_url=db_url, model_loader=fake_loader)
        with TestClient(app) as tc:
            tc.db_url = db_url  # type: ignore
            yield tc


def test_auth_and_signup_flow(client: TestClient) -> None:
    # 1. Signup then /me
    res = client.post("/signup", json={"username": "alice", "password": "password123"})
    assert res.status_code == 201
    data = res.json()
    assert data["username"] == "alice"
    assert data["user_id"] >= 100000
    token = data["token"]

    me_res = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["username"] == "alice"

    # 2. Duplicate username returns 409
    dup = client.post("/signup", json={"username": "alice", "password": "password123"})
    assert dup.status_code == 409

    # 3. Wrong password returns 401
    wrong_pw = client.post("/login", json={"username": "alice", "password": "wrongpassword"})
    assert wrong_pw.status_code == 401

    # 4. Weak password returns 422
    weak_pw = client.post("/signup", json={"username": "bob_user", "password": "123"})
    assert weak_pw.status_code == 422


def test_popular_and_recommend_flow(client: TestClient) -> None:
    # 5. /popular works for a guest
    pop = client.get("/popular")
    assert pop.status_code == 200
    assert pop.json()["kind"] == "popular"
    assert len(pop.json()["movies"]) > 0

    # 6. /feedback without a token returns 401
    fb_no_auth = client.post("/feedback", json={"movie_id": 10, "rating": 4.0})
    assert fb_no_auth.status_code == 401

    # Sign up new user
    auth = client.post("/signup", json={"username": "carol", "password": "password123"}).json()
    headers = {"Authorization": f"Bearer {auth['token']}"}

    # 7. New user gets kind=popular from /recommend
    rec_init = client.get("/recommend", headers=headers)
    assert rec_init.status_code == 200
    assert rec_init.json()["kind"] == "popular"

    # 8. After 3 ratings /recommend returns personalized and never contains rated movies
    client.post("/feedback", json={"movie_id": 10, "rating": 5.0}, headers=headers)
    client.post("/feedback", json={"movie_id": 20, "rating": 4.0}, headers=headers)
    client.post("/feedback", json={"movie_id": 30, "rating": 4.5}, headers=headers)

    rec_personalized = client.get("/recommend", headers=headers)
    assert rec_personalized.status_code == 200
    assert rec_personalized.json()["kind"] == "personalized"
    rec_ids = [m["movie_id"] for m in rec_personalized.json()["movies"]]
    assert 10 not in rec_ids
    assert 20 not in rec_ids
    assert 30 not in rec_ids

    # 9. After one 4-star rating /because has a row
    bec = client.get("/because", headers=headers)
    assert bec.status_code == 200
    assert len(bec.json()["rows"]) >= 1

    # 10. Rating 0.7 returns 422
    invalid_r = client.post("/feedback", json={"movie_id": 40, "rating": 0.7}, headers=headers)
    assert invalid_r.status_code == 422

    # 11. Unknown movie returns 404
    unknown_m = client.post("/feedback", json={"movie_id": 999999, "rating": 4.0}, headers=headers)
    assert unknown_m.status_code == 404


def test_simulator_and_admin_security(client: TestClient) -> None:
    # 12. Simulator key + user_id stores source=simulator while normal call stores web
    engine = get_engine(client.db_url)  # type: ignore

    sim_headers = {"X-Simulator-Key": "sim-secret-change-me"}
    sim_fb = client.post(
        "/feedback",
        json={"movie_id": 10, "rating": 4.0, "user_id": 15},
        headers=sim_headers,
    )
    assert sim_fb.status_code == 200

    session = next(get_session(engine))
    row = session.query(Feedback).filter(Feedback.user_id == 15).first()
    assert row is not None
    assert row.source == "simulator"

    # 13. Simulator key with user_id 100001 returns 403
    sim_invalid_uid = client.post(
        "/feedback",
        json={"movie_id": 10, "rating": 4.0, "user_id": 100001},
        headers=sim_headers,
    )
    assert sim_invalid_uid.status_code == 403

    # 14. Wrong simulator key returns 403
    sim_wrong_key = client.post(
        "/feedback",
        json={"movie_id": 10, "rating": 4.0, "user_id": 15},
        headers={"X-Simulator-Key": "wrong-key"},
    )
    assert sim_wrong_key.status_code == 403

    # 15. /reload without admin key returns 403
    reload_res = client.post("/reload", headers={"X-Admin-Key": "wrong-admin-key"})
    assert reload_res.status_code == 403

    # Reload with correct admin key returns 200
    reload_ok = client.post("/reload", headers={"X-Admin-Key": "admin-secret-change-me"})
    assert reload_ok.status_code == 200
    assert reload_ok.json()["model_version"] == 1

    # 16. /health shows replica name and version
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["model_version"] == 1
    assert "replica" in health.json()

    # 17. Response has X-Served-By header
    assert "x-served-by" in health.headers
