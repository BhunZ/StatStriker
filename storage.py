"""
storage.py
----------
Persistence layer for pipeline state and model artefacts.

Supports two backends:
- ``LocalStorage``  : reads/writes to local disk (default, for dev)
- ``S3Storage``     : reads/writes to an S3 bucket (for cloud CI/CD)

Usage
-----
    # Local (default)
    store = StorageManager.create("local", base_dir="data/models")

    # S3
    store = StorageManager.create(
        "s3",
        bucket="my-pl-models",
        prefix="prod/v1",
    )

    # Uniform API
    state = store.load_state()
    store.save_state(state)
    store.upload_model("dixon_coles.pkl", local_path)
    store.download_model("dixon_coles.pkl", local_path)
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATE_FILENAME = "pipeline_state.json"

# ---------------------------------------------------------------------------
# Default state (first run)
# ---------------------------------------------------------------------------

_DEFAULT_STATE: dict[str, Any] = {
    "last_scrape_date": None,
    "last_scrape_matches": 0,
    "last_train_date": None,
    "last_train_matches": 0,
    "last_train_log_loss": None,
    "model_version": 0,
    "recent_predictions": [],
}


# ---------------------------------------------------------------------------
# JSON encoder for dates
# ---------------------------------------------------------------------------

class _DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class StorageBackend(ABC):
    """Abstract interface for pipeline persistence."""

    @abstractmethod
    def load_state(self) -> dict[str, Any]:
        """Load pipeline_state.json. Return default state if not found."""

    @abstractmethod
    def save_state(self, state: dict[str, Any]) -> None:
        """Write pipeline_state.json."""

    @abstractmethod
    def upload_model(self, name: str, local_path: Path) -> None:
        """Push a model .pkl to the storage backend."""

    @abstractmethod
    def download_model(self, name: str, local_path: Path) -> None:
        """Pull a model .pkl from the storage backend."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """List available model file names."""


# ---------------------------------------------------------------------------
# Local filesystem backend
# ---------------------------------------------------------------------------

class LocalStorage(StorageBackend):
    """Store state and models on the local filesystem."""

    def __init__(self, base_dir: str | Path) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _state_path(self) -> Path:
        return self._base / STATE_FILENAME

    def load_state(self) -> dict[str, Any]:
        path = self._state_path()
        if not path.exists():
            logger.info("No pipeline state found at %s — using defaults", path)
            return _DEFAULT_STATE.copy()
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        logger.info(
            "Loaded pipeline state: v%s, last_train=%s, train_matches=%d",
            state.get("model_version", "?"),
            state.get("last_train_date", "never"),
            state.get("last_train_matches", 0),
        )
        return state

    def save_state(self, state: dict[str, Any]) -> None:
        path = self._state_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, cls=_DateEncoder)
        logger.info("Pipeline state saved to %s (v%s)", path, state.get("model_version"))

    def upload_model(self, name: str, local_path: Path) -> None:
        import shutil
        dest = self._base / name
        if Path(local_path) != dest:
            shutil.copy2(local_path, dest)
        logger.debug("Model stored locally: %s", dest)

    def download_model(self, name: str, local_path: Path) -> None:
        import shutil
        src = self._base / name
        if not src.exists():
            raise FileNotFoundError(f"Model not found: {src}")
        if src != Path(local_path):
            shutil.copy2(src, local_path)
        logger.debug("Model loaded locally: %s", src)

    def list_models(self) -> list[str]:
        return sorted(p.name for p in self._base.glob("*.pkl"))


# ---------------------------------------------------------------------------
# S3 backend
# ---------------------------------------------------------------------------

class S3Storage(StorageBackend):
    """
    Store state and models in an AWS S3 bucket.

    Requires ``boto3`` and valid AWS credentials (env vars, IAM role, or
    ~/.aws/credentials).

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    prefix : str
        Key prefix (e.g. "prod/v1"). No trailing slash.
    """

    def __init__(self, bucket: str, prefix: str = "") -> None:
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3Storage. Install with: pip install boto3"
            )
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._s3 = boto3.client("s3")
        logger.info("S3Storage initialised: s3://%s/%s/", bucket, self._prefix)

    def _key(self, name: str) -> str:
        return f"{self._prefix}/{name}" if self._prefix else name

    def load_state(self) -> dict[str, Any]:
        import botocore.exceptions
        try:
            resp = self._s3.get_object(
                Bucket=self._bucket, Key=self._key(STATE_FILENAME)
            )
            state = json.loads(resp["Body"].read().decode("utf-8"))
            logger.info(
                "Loaded pipeline state from S3: v%s, train_matches=%d",
                state.get("model_version", "?"),
                state.get("last_train_matches", 0),
            )
            return state
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.info("No pipeline state in S3 — using defaults")
                return _DEFAULT_STATE.copy()
            raise

    def save_state(self, state: dict[str, Any]) -> None:
        body = json.dumps(state, indent=2, cls=_DateEncoder)
        self._s3.put_object(
            Bucket=self._bucket,
            Key=self._key(STATE_FILENAME),
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(
            "Pipeline state saved to s3://%s/%s (v%s)",
            self._bucket, self._key(STATE_FILENAME),
            state.get("model_version"),
        )

    def upload_model(self, name: str, local_path: Path) -> None:
        self._s3.upload_file(str(local_path), self._bucket, self._key(name))
        logger.info("Model uploaded: s3://%s/%s", self._bucket, self._key(name))

    def download_model(self, name: str, local_path: Path) -> None:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(self._bucket, self._key(name), str(local_path))
        logger.info("Model downloaded: s3://%s/%s -> %s", self._bucket, self._key(name), local_path)

    def list_models(self) -> list[str]:
        import botocore.exceptions
        try:
            prefix = self._key("")
            resp = self._s3.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
            return sorted(
                obj["Key"].split("/")[-1]
                for obj in resp.get("Contents", [])
                if obj["Key"].endswith(".pkl")
            )
        except botocore.exceptions.ClientError:
            return []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class StorageManager:
    """
    Factory for creating the appropriate storage backend.

    Usage
    -----
        store = StorageManager.create("local", base_dir="data/models")
        store = StorageManager.create("s3", bucket="my-bucket", prefix="prod")
    """

    @staticmethod
    def create(backend: str = "local", **kwargs) -> StorageBackend:
        """
        Create a storage backend.

        Parameters
        ----------
        backend : str
            "local" or "s3"
        **kwargs :
            For "local": ``base_dir`` (str | Path)
            For "s3": ``bucket`` (str), ``prefix`` (str, optional)
        """
        if backend == "local":
            base_dir = kwargs.get("base_dir", "data/models")
            return LocalStorage(base_dir)
        elif backend == "s3":
            bucket = kwargs["bucket"]
            prefix = kwargs.get("prefix", "")
            return S3Storage(bucket, prefix)
        else:
            raise ValueError(f"Unknown storage backend: {backend!r}. Use 'local' or 's3'.")
