"""Authentication, password hashing, and session management for the API."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Header, HTTPException, status
from sqlalchemy.orm import Session

from src.common.config import get_settings
from src.common.db import SessionModel, User

USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_]{3,30}$")


def validate_credentials(username: str, password: str) -> str:
    """Validate username and password constraints. Returns normalized lowercase username."""
    clean_username = username.strip().lower()
    if not USERNAME_REGEX.match(clean_username):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Username must be 3-30 characters (letters, digits, underscores).",
        )
    if not (6 <= len(password) <= 128):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must be between 6 and 128 characters.",
        )
    return clean_username


def hash_password(password: str) -> str:
    """Hash password using scrypt (n=16384, r=8, p=1) and a 16-byte random salt."""
    salt = os.urandom(16)
    pw_hash = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, maxmem=0)
    return f"{salt.hex()}${pw_hash.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored salt$hash string using constant-time comparison."""
    try:
        salt_hex, hash_hex = stored_hash.split("$")
        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(hash_hex)
        computed_hash = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, maxmem=0
        )
        return hmac.compare_digest(computed_hash, expected_hash)
    except Exception:
        return False


def hash_token(token: str) -> str:
    """Compute SHA-256 hex digest of session token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(db: Session, user_id: int) -> str:
    """Generate 7-day session token and store SHA-256 hash in database."""
    token = secrets.token_urlsafe(32)
    token_h = hash_token(token)
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)

    session_record = SessionModel(
        token_hash=token_h,
        user_id=user_id,
        created_at=datetime.now(timezone.utc),
        expires_at=expires_at,
    )
    db.add(session_record)
    db.commit()
    return token


def get_token_from_header(authorization: str | None = Header(None)) -> str | None:
    """Extract Bearer token from Authorization header if present."""
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def optional_user(
    db: Session,
    authorization: str | None = None,
) -> User | None:
    """Return authenticated User if Bearer token is valid and unexpired, else None."""
    token = get_token_from_header(authorization)
    if not token:
        return None

    token_h = hash_token(token)
    session_record = db.query(SessionModel).filter(SessionModel.token_hash == token_h).first()
    if not session_record:
        return None

    now = datetime.now(timezone.utc)
    # Handle both timezone-aware and timezone-naive expires_at from SQLite
    expires_at = session_record.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if now > expires_at:
        db.delete(session_record)
        db.commit()
        return None

    return db.query(User).filter(User.id == session_record.user_id).first()


def current_user(
    db: Session,
    authorization: str | None = None,
) -> User:
    """Require valid authenticated User, raising 401 if missing or expired."""
    user = optional_user(db=db, authorization=authorization)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or expired session token",
        )
    return user


def check_simulator_key(x_simulator_key: str | None = None) -> str:
    """Verify simulator header key, returning 'simulator', 'web', or raising 403."""
    if x_simulator_key is None:
        return "web"

    settings = get_settings()
    if x_simulator_key == settings.simulator_key:
        return "simulator"

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid X-Simulator-Key header",
    )
