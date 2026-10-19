"""Tests for retrain logic pure functions."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from src.train.retrain import (
    decide_promotion,
    merge_training_data,
    should_trigger,
    split_holdout,
)

# ---------------------------------------------------------------------------
# should_trigger() tests
# ---------------------------------------------------------------------------


class TestShouldTrigger:
    PARAMS = {"retrain": {"min_new_rows": 500}}

    def _make_drift_row(self, version: int, detected: bool) -> SimpleNamespace:
        return SimpleNamespace(model_version=version, drift_detected=detected)

    def test_no_drift_row_returns_false(self):
        assert should_trigger(None, 1, 600, self.PARAMS) is False

    def test_drift_off_returns_false(self):
        row = self._make_drift_row(1, False)
        assert should_trigger(row, 1, 600, self.PARAMS) is False

    def test_drift_on_but_too_few_rows(self):
        row = self._make_drift_row(1, True)
        assert should_trigger(row, 1, 499, self.PARAMS) is False

    def test_drift_on_and_enough_rows(self):
        row = self._make_drift_row(1, True)
        assert should_trigger(row, 1, 500, self.PARAMS) is True

    def test_drift_row_from_older_champion(self):
        """Drift row belongs to older champion version -> no trigger."""
        row = self._make_drift_row(1, True)  # version 1
        assert should_trigger(row, 2, 1000, self.PARAMS) is False  # champion is now 2

    def test_exactly_at_min_rows(self):
        row = self._make_drift_row(1, True)
        assert should_trigger(row, 1, 500, self.PARAMS) is True


# ---------------------------------------------------------------------------
# split_holdout() tests
# ---------------------------------------------------------------------------


class TestSplitHoldout:
    def _make_df(self, n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "userId": list(range(n)),
                "movieId": list(range(n)),
                "rating": [4.0] * n,
                "timestamp": list(range(n)),  # timestamps 0..n-1
            }
        )

    def test_newest_20_percent_in_holdout(self):
        df = self._make_df(100)
        older, holdout = split_holdout(df, 0.2)
        assert len(holdout) == 20
        assert len(older) == 80

    def test_holdout_has_largest_timestamps(self):
        df = self._make_df(10)
        _, holdout = split_holdout(df, 0.3)
        # The holdout should be the 3 newest rows (timestamps 7,8,9)
        assert holdout["timestamp"].min() >= 7

    def test_empty_df_returns_empty(self):
        empty = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
        older, holdout = split_holdout(empty, 0.2)
        assert older.empty
        assert holdout.empty


# ---------------------------------------------------------------------------
# decide_promotion() tests
# ---------------------------------------------------------------------------


class TestDecidePromotion:
    PARAMS = {
        "retrain": {
            "min_improvement": 0.01,
            "max_test_drop": 0.10,
        }
    }

    def test_1_percent_gain_passes(self):
        # 1% gain on holdout, no test drop -> promote
        promote, _ = decide_promotion(0.10, 0.101, 0.20, 0.20, self.PARAMS)
        assert promote is True

    def test_0_9_percent_gain_fails(self):
        # <1% gain -> reject
        promote, _ = decide_promotion(0.10, 0.1009, 0.20, 0.20, self.PARAMS)
        assert promote is False

    def test_10_percent_test_drop_passes(self):
        # 10% drop on test is at the limit -> still acceptable
        v1_test = 0.20
        v2_test = v1_test * (1 - 0.10)
        promote, _ = decide_promotion(0.10, 0.11, v1_test, v2_test, self.PARAMS)
        assert promote is True

    def test_11_percent_test_drop_fails(self):
        # >10% drop on test -> reject
        v1_test = 0.20
        v2_test = v1_test * (1 - 0.11)
        promote, _ = decide_promotion(0.10, 0.11, v1_test, v2_test, self.PARAMS)
        assert promote is False

    def test_both_conditions_must_hold(self):
        # Good gain but bad test drop -> reject
        v1_test = 0.20
        v2_test = v1_test * (1 - 0.15)  # 15% drop
        promote, _ = decide_promotion(0.10, 0.15, v1_test, v2_test, self.PARAMS)
        assert promote is False


# ---------------------------------------------------------------------------
# merge_training_data() tests
# ---------------------------------------------------------------------------


class TestMergeTrainingData:
    def _make_df(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_latest_rating_wins(self):
        train = self._make_df([{"userId": 1, "movieId": 100, "rating": 3.0, "timestamp": 1000}])
        feedback = self._make_df([{"userId": 1, "movieId": 100, "rating": 5.0, "timestamp": 2000}])
        merged = merge_training_data(train, feedback)
        row = merged[(merged["userId"] == 1) & (merged["movieId"] == 100)]
        assert len(row) == 1
        assert row["rating"].iloc[0] == 5.0

    def test_combines_both_sources(self):
        train = self._make_df([{"userId": 1, "movieId": 10, "rating": 4.0, "timestamp": 1000}])
        feedback = self._make_df([{"userId": 2, "movieId": 20, "rating": 3.0, "timestamp": 2000}])
        merged = merge_training_data(train, feedback)
        assert len(merged) == 2

    def test_unique_per_user_movie(self):
        train = self._make_df(
            [
                {"userId": 1, "movieId": 10, "rating": 2.0, "timestamp": 100},
                {"userId": 1, "movieId": 10, "rating": 3.0, "timestamp": 200},
            ]
        )
        merged = merge_training_data(train, pd.DataFrame())
        row = merged[(merged["userId"] == 1) & (merged["movieId"] == 10)]
        assert len(row) == 1
        assert row["rating"].iloc[0] == 3.0


# ---------------------------------------------------------------------------
# reload_replicas fault isolation test
# ---------------------------------------------------------------------------


class TestReloadReplicas:
    """reload_replicas still calls the second target when the first one fails."""

    def test_continues_after_first_failure(self):
        call_log: list[str] = []

        def fake_post(url: str, headers=None, timeout=None):
            call_log.append(url)
            if "api1" in url:
                raise ConnectionError("api1 down")
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"model_version": 2}
            return resp

        with patch("src.common.reload.requests.post", side_effect=fake_post):
            with patch(
                "src.common.reload.get_settings",
                return_value=SimpleNamespace(
                    admin_key="test-key",
                    api_reload_targets=["http://api1:8000", "http://api2:8000"],
                    api_headless_host=None,
                ),
            ):
                from src.common.reload import reload_replicas

                results = reload_replicas()

        assert len(results) == 2
        targets = [r[0] for r in results]
        assert any("api1" in t for t in targets)
        assert any("api2" in t for t in targets)

        # api1 failed, api2 succeeded
        api1_result = next(r for r in results if "api1" in r[0])
        api2_result = next(r for r in results if "api2" in r[0])
        assert api1_result[1] is False
        assert api2_result[1] is True
        assert api2_result[2] == 2

        # Both were called
        assert len(call_log) == 2
