"""FastAPI application serving movie recommendations, auth, feedback, and metrics."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.auth import (
    check_simulator_key,
    create_session,
    current_user,
    hash_password,
    hash_token,
    optional_user,
    validate_credentials,
    verify_password,
)
from src.api.metrics import (
    PrometheusMetricsMiddleware,
    metrics_response,
    record_feedback,
    record_recommendation,
    set_model_version,
)
from src.api.state import LoadedModel, ModelHolder, load_champion
from src.common.config import Settings, get_settings, load_params
from src.common.db import (
    Feedback,
    RecommendationLog,
    SessionModel,
    User,
    get_engine,
    get_session,
    init_db,
)
from src.recsys.bundle import Bundle
from src.recsys.recommend import because_rows, recommend_for_user, recommend_popular


class SignupLoginRequest(BaseModel):
    username: str
    password: str


class FeedbackRequest(BaseModel):
    movie_id: int
    rating: float
    user_id: int | None = None


class XServedByMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, replica_name: str) -> None:
        super().__init__(app)
        self.replica_name = replica_name

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Served-By"] = self.replica_name
        return response


def get_merged_history(bundle: Bundle, db: Session, user_id: int) -> pd.DataFrame:
    """Merge bundle training history with database feedback, keeping latest rating per movie."""
    bundle_h = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
    if user_id in bundle.user_to_idx:
        bh = bundle.user_history[bundle.user_history["userId"] == user_id]
        if len(bh) > 0:
            bundle_h = bh.copy()

    fb_rows = db.query(Feedback).filter(Feedback.user_id == user_id).all()
    if not fb_rows:
        return bundle_h

    db_records = []
    for f in fb_rows:
        ts = int(f.created_at.timestamp()) if f.created_at else 0
        db_records.append(
            {
                "userId": user_id,
                "movieId": int(f.movie_id),
                "rating": float(f.rating),
                "timestamp": ts,
            }
        )
    fb_df = pd.DataFrame(db_records)

    combined = pd.concat([bundle_h, fb_df], ignore_index=True)
    merged = (
        combined.sort_values(by="timestamp")
        .drop_duplicates(subset=["movieId"], keep="last")
        .reset_index(drop=True)
    )
    return merged


def create_app(
    database_url: str | None = None,
    model_loader: Callable[[Settings], LoadedModel] | None = None,
) -> FastAPI:
    """Application factory for FastAPI serving service."""
    settings = get_settings()
    db_url = database_url or settings.database_url
    engine = get_engine(db_url)
    model_holder = ModelHolder()
    params = load_params()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(engine)
        loader = model_loader or load_champion
        try:
            initial_model = loader(settings)
            model_holder.swap(initial_model)
            set_model_version(initial_model.version)
        except Exception as e:
            raise RuntimeError(f"No champion model in MLflow. Run the trainer first: {e}") from e
        yield

    app = FastAPI(title="Movie Recommender API", lifespan=lifespan)

    app.add_middleware(XServedByMiddleware, replica_name=settings.replica_name)
    app.add_middleware(PrometheusMetricsMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def db_dep() -> Session:
        yield from get_session(engine)

    def get_bundle() -> tuple[Bundle, int]:
        m = model_holder.get()
        if m is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )
        return m.bundle, m.version

    @app.get("/health")
    def health() -> dict[str, Any]:
        m = model_holder.get()
        if m is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )
        return {
            "status": "ok",
            "model_version": m.version,
            "replica": settings.replica_name,
        }

    @app.get("/metrics")
    def metrics() -> Response:
        return metrics_response()

    @app.post("/reload")
    def reload_model(x_admin_key: str | None = Header(None)) -> dict[str, Any]:
        if x_admin_key != settings.admin_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid admin key",
            )
        loader = model_loader or load_champion
        try:
            new_model = loader(settings)
            model_holder.swap(new_model)
            set_model_version(new_model.version)
            return {"model_version": new_model.version}
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Reload failed: {e}",
            )

    @app.post("/signup", status_code=status.HTTP_201_CREATED)
    def signup(req: SignupLoginRequest, db: Session = Depends(db_dep)) -> dict[str, Any]:
        username = validate_credentials(req.username, req.password)
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already exists",
            )
        user = User(username=username, password_hash=hash_password(req.password))
        db.add(user)
        db.commit()
        db.refresh(user)

        token = create_session(db, user.id)
        return {"user_id": user.public_id, "username": user.username, "token": token}

    @app.post("/login")
    def login(req: SignupLoginRequest, db: Session = Depends(db_dep)) -> dict[str, Any]:
        username = req.username.strip().lower()
        user = db.query(User).filter(User.username == username).first()
        if not user or not verify_password(req.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )
        token = create_session(db, user.id)
        return {"user_id": user.public_id, "username": user.username, "token": token}

    @app.post("/logout")
    def logout(
        authorization: str | None = Header(None),
        db: Session = Depends(db_dep),
    ) -> dict[str, bool]:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Bearer token",
            )
        token = authorization.split()[1]
        token_h = hash_token(token)
        sess = db.query(SessionModel).filter(SessionModel.token_hash == token_h).first()
        if sess:
            db.delete(sess)
            db.commit()
        return {"ok": True}

    @app.get("/me")
    def get_me(
        authorization: str | None = Header(None),
        db: Session = Depends(db_dep),
    ) -> dict[str, Any]:
        user = current_user(db=db, authorization=authorization)
        bundle, _ = get_bundle()
        history = get_merged_history(bundle, db, user.public_id)
        n_ratings = int(history["movieId"].nunique()) if len(history) > 0 else 0
        return {
            "user_id": user.public_id,
            "username": user.username,
            "n_ratings": n_ratings,
        }

    @app.get("/popular")
    def get_popular(
        n: int = Query(default=10, ge=1, le=50),
        authorization: str | None = Header(None),
        db: Session = Depends(db_dep),
    ) -> dict[str, Any]:
        bundle, version = get_bundle()
        seen: set[int] = set()
        user = optional_user(db=db, authorization=authorization)
        if user:
            history = get_merged_history(bundle, db, user.public_id)
            seen = set(history["movieId"].astype(int))

        movies = recommend_popular(bundle, seen, n)
        return {
            "kind": "popular",
            "model_version": version,
            "movies": movies,
        }

    @app.get("/recommend")
    def get_recommend(
        n: int = Query(default=10, ge=1, le=50),
        user_id: int | None = Query(default=None),
        authorization: str | None = Header(None),
        x_simulator_key: str | None = Header(None),
        db: Session = Depends(db_dep),
    ) -> dict[str, Any]:
        bundle, version = get_bundle()
        source = check_simulator_key(x_simulator_key)

        effective_uid: int
        if source == "simulator":
            if user_id is None or not (1 <= user_id <= 99999):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Simulator user_id must be between 1 and 99999",
                )
            effective_uid = user_id
        else:
            user = current_user(db=db, authorization=authorization)
            effective_uid = user.public_id

        history = get_merged_history(bundle, db, effective_uid)
        kind, movies = recommend_for_user(bundle, history, n)

        # Log recommendation
        movie_ids = [m["movie_id"] for m in movies]
        scores = [m["score"] for m in movies]
        log = RecommendationLog(
            user_id=effective_uid,
            kind=kind,
            model_version=version,
            movie_ids=json.dumps(movie_ids),
            scores=json.dumps(scores),
            source=source,
        )
        db.add(log)
        db.commit()

        record_recommendation(kind)
        return {"kind": kind, "model_version": version, "movies": movies}

    @app.get("/because")
    def get_because(
        n: int = Query(default=10, ge=1, le=50),
        user_id: int | None = Query(default=None),
        authorization: str | None = Header(None),
        x_simulator_key: str | None = Header(None),
        db: Session = Depends(db_dep),
    ) -> dict[str, Any]:
        bundle, version = get_bundle()
        source = check_simulator_key(x_simulator_key)

        effective_uid: int
        if source == "simulator":
            if user_id is None or not (1 <= user_id <= 99999):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Simulator user_id must be between 1 and 99999",
                )
            effective_uid = user_id
        else:
            user = current_user(db=db, authorization=authorization)
            effective_uid = user.public_id

        history = get_merged_history(bundle, db, effective_uid)
        rows = because_rows(bundle, history, n)

        all_rec_ids = [m["movie_id"] for r in rows for m in r["movies"]]
        all_rec_scores = [m["score"] for r in rows for m in r["movies"]]
        log = RecommendationLog(
            user_id=effective_uid,
            kind="because",
            model_version=version,
            movie_ids=json.dumps(all_rec_ids),
            scores=json.dumps(all_rec_scores),
            source=source,
        )
        db.add(log)
        db.commit()

        record_recommendation("because")
        return {"model_version": version, "rows": rows}

    @app.get("/preview/{target_user_id}")
    def preview_user(target_user_id: int) -> dict[str, Any]:
        if not (1 <= target_user_id <= 99999):
            raise HTTPException(status_code=404, detail="Invalid MovieLens user id")

        bundle, _ = get_bundle()
        if target_user_id not in bundle.user_to_idx:
            raise HTTPException(status_code=404, detail="User not found in model")

        bh = bundle.user_history[bundle.user_history["userId"] == target_user_id]
        n_default = int(params.get("serving", {}).get("default_n", 10))

        kind, rec_movies = recommend_for_user(bundle, bh, n_default)
        bec_rows = because_rows(bundle, bh, n_default)

        return {
            "recommended": {"kind": kind, "movies": rec_movies},
            "because": {"rows": bec_rows},
        }

    @app.post("/feedback")
    def submit_feedback(
        req: FeedbackRequest,
        authorization: str | None = Header(None),
        x_simulator_key: str | None = Header(None),
        db: Session = Depends(db_dep),
    ) -> dict[str, bool]:
        # Validate rating
        valid_ratings = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}
        if req.rating not in valid_ratings:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Rating must be 0.5 to 5.0 in steps of 0.5",
            )

        bundle, _ = get_bundle()
        if req.movie_id not in bundle.movie_meta:
            raise HTTPException(status_code=404, detail="Unknown movie_id")

        source = check_simulator_key(x_simulator_key)
        effective_uid: int
        if source == "simulator":
            if req.user_id is None or not (1 <= req.user_id <= 99999):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Simulator user_id must be between 1 and 99999",
                )
            effective_uid = req.user_id
        else:
            user = current_user(db=db, authorization=authorization)
            effective_uid = user.public_id

        fb = Feedback(
            user_id=effective_uid,
            movie_id=req.movie_id,
            rating=req.rating,
            source=source,
        )
        db.add(fb)
        db.commit()

        record_feedback(source=source, rating=req.rating)
        return {"ok": True}

    @app.get("/")
    def serve_index() -> FileResponse:
        index_file = Path(__file__).parent / "static" / "index.html"
        return FileResponse(index_file)

    return app


app = create_app()
