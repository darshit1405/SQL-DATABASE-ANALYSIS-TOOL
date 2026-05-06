"""
Central app configuration.
Loads values from .env and gives helper methods.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
from dotenv import dotenv_values, load_dotenv


# Always load .env from this project folder (reliable in Streamlit reruns).
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)


def _parse_env_text(content: str) -> Dict[str, str]:
    """
    Parse simple KEY=VALUE lines from .env content.
    Handles comments and quoted values.
    """
    parsed: Dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")  # remove optional quotes
        if key:
            parsed[key] = value
    return parsed


def _load_dotenv_values_flexible(path: Path) -> Dict[str, str]:
    """
    Read .env with multiple encodings to avoid Windows encoding issues.
    """
    # Try python-dotenv first.
    values = {k: v for k, v in dotenv_values(path).items() if v is not None}
    if values:
        return values

    # Fallback: manual parse using common encodings.
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            text = path.read_text(encoding=encoding)
        except Exception:
            continue
        parsed = _parse_env_text(text)
        if parsed:
            return parsed

    return {}


DOTENV_VALUES = _load_dotenv_values_flexible(ENV_PATH)


def _env(name: str, default: str = "") -> str:
    """
    Read from process env first, then fallback to parsed .env map.
    """
    value = os.getenv(name)
    if value is None or value == "":
        value = DOTENV_VALUES.get(name, default)
    return str(value).strip() if value is not None else default


@dataclass
class Settings:
    """Simple settings object for app-wide use."""

    # Use default_factory so values are read at object creation time.
    db_type: str = field(default_factory=lambda: _env("DB_TYPE", "mysql").lower())
    db_host: str = field(default_factory=lambda: _env("DB_HOST", "localhost"))
    db_port: int = field(default_factory=lambda: int(_env("DB_PORT", "3306")))
    db_name: str = field(default_factory=lambda: _env("DB_NAME", ""))
    db_user: str = field(default_factory=lambda: _env("DB_USER", ""))
    db_password: str = field(default_factory=lambda: _env("DB_PASSWORD", ""))
    together_api_key: str = field(default_factory=lambda: _env("TOGETHER_API_KEY", ""))
    together_model: str = field(
        default_factory=lambda: _env(
            "TOGETHER_MODEL",
            "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        )
    )

    def validate(self) -> None:
        """
        Check required values early.
        Raises ValueError with clear messages.
        """
        if self.db_type not in {"mysql", "postgresql", "sqlite"}:
            raise ValueError("DB_TYPE must be 'mysql', 'postgresql', or 'sqlite'.")

        if self.db_type != "sqlite":
            required_fields = {
                "DB_HOST": self.db_host,
                "DB_PORT": str(self.db_port),
                "DB_NAME": self.db_name,
                "DB_USER": self.db_user,
            }
            missing = [name for name, value in required_fields.items() if not value]
            if missing:
                raise ValueError(f"Missing required database settings: {', '.join(missing)}")

    def sqlalchemy_url(self) -> str:
        """
        Build SQLAlchemy connection URL for selected DB.
        """
        if self.db_type == "sqlite":
            # For sqlite, DB_NAME is the file path (e.g. database.db), or empty for memory
            return f"sqlite:///{self.db_name}"

        if self.db_type == "mysql":
            driver = "mysql+pymysql"
        else:
            driver = "postgresql+psycopg2"

        return (
            f"{driver}://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


def get_settings() -> Settings:
    """
    Return validated settings object.
    Use this function in other files.
    """
    settings = Settings()
    settings.validate()
    return settings