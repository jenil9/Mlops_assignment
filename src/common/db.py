"""SQLAlchemy database models and connection helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Generator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()


def utc_now() -> datetime:
    """Return current UTC timestamp with timezone."""
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    @property
    def public_id(self) -> int:
        return 100000 + int(self.id)


class SessionModel(Base):
    __tablename__ = "sessions"

    token_hash = Column(String(64), primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    movie_id = Column(Integer, nullable=False)
    rating = Column(Float, nullable=False)
    source = Column(String(20), nullable=False)  # 'web' or 'simulator'
    created_at = Column(DateTime(timezone=True), default=utc_now, index=True, nullable=False)


class RecommendationLog(Base):
    __tablename__ = "recommendation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    kind = Column(String(30), nullable=False)  # 'popular', 'personalized', 'because'
    model_version = Column(Integer, nullable=False)
    movie_ids = Column(Text, nullable=False)  # JSON text
    scores = Column(Text, nullable=False)  # JSON text
    source = Column(String(20), nullable=False)  # 'web' or 'simulator'
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class DriftReport(Base):
    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    model_version = Column(Integer, nullable=False)
    n_rows = Column(Integer, nullable=False)
    rows_by_source = Column(Text, nullable=False)  # JSON text
    drift_share = Column(Float, nullable=True)
    drift_detected = Column(Boolean, nullable=False)
    live_hit_rate = Column(Float, nullable=True)
    report_path = Column(String(255), nullable=True)
    note = Column(Text, nullable=True)


class RetrainRun(Base):
    __tablename__ = "retrain_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    started_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    trigger = Column(String(30), nullable=False)  # 'drift', 'manual', 'forced'
    status = Column(String(30), nullable=False)  # 'running', 'done', 'failed'
    rows_used = Column(Integer, nullable=True)
    holdout_rows = Column(Integer, nullable=True)
    v1_version = Column(Integer, nullable=True)
    v2_version = Column(Integer, nullable=True)
    v1_ndcg_holdout = Column(Float, nullable=True)
    v2_ndcg_holdout = Column(Float, nullable=True)
    v1_ndcg_test = Column(Float, nullable=True)
    v2_ndcg_test = Column(Float, nullable=True)
    decision = Column(String(30), nullable=True)  # 'promoted', 'rejected', 'forced', 'failed'
    note = Column(Text, nullable=True)


def get_engine(url: str) -> Engine:
    """Create SQLAlchemy engine with connection verification."""
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)


def init_db(engine: Engine) -> None:
    """Initialize database schema if tables do not exist."""
    Base.metadata.create_all(engine)


def get_session(engine: Engine) -> Generator[Session, None, None]:
    """Yield database session and guarantee closure."""
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()
