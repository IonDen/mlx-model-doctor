"""File-based JSON cache for HF listing responses."""

import hashlib
import json
import time
from pathlib import Path

CacheKey = tuple[str, str | None, int]
_DEFAULT_CACHE_PATH = ".cache/mlx-model-doctor/hf-listings"


def default_cache_dir() -> Path:
    """Return the default cache directory (~/.cache/mlx-model-doctor/hf-listings/)."""
    return Path.home() / _DEFAULT_CACHE_PATH


class ListingCache:
    """Simple file-based JSON cache with TTL expiry."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        ttl_seconds: int = 3600,
    ) -> None:
        """Initialize the cache with an optional cache directory and TTL.

        Args:
            cache_dir: Directory to store cache files. Defaults to ~/.cache/mlx-model-doctor/hf-listings/.
            ttl_seconds: Time-to-live for cached entries in seconds.
        """
        self._cache_dir = cache_dir if cache_dir is not None else default_cache_dir()
        self._ttl_seconds = ttl_seconds

    def get(self, key: CacheKey) -> list[dict[str, object]] | None:
        """Retrieve a cached listing by key.

        Args:
            key: A (author, task, max_candidates) tuple.

        Returns:
            The cached data if found and not expired, None otherwise.
        """
        path = self._path_for(key)
        try:
            if not path.exists():
                return None
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(raw, dict):
            return None
        timestamp = raw.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            return None
        if time.time() - timestamp > self._ttl_seconds:
            return None
        data = raw.get("data")
        if not isinstance(data, list):
            return None
        if not all(isinstance(item, dict) and "repo_id" in item for item in data):
            return None
        return data

    def put(self, key: CacheKey, data: list[dict[str, object]]) -> None:
        """Store a listing in the cache.

        Args:
            key: A (author, task, max_candidates) tuple.
            data: The listing data to cache.
        """
        path = self._path_for(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"timestamp": time.time(), "data": data}
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        except OSError:
            pass

    def clear(self) -> None:
        """Clear all cached entries."""
        if self._cache_dir.exists():
            for f in self._cache_dir.iterdir():
                if f.suffix == ".json":
                    f.unlink(missing_ok=True)

    def _path_for(self, key: CacheKey) -> Path:
        author, task, max_candidates = key
        raw = f"{author}:{json.dumps(task)}:{max_candidates}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return self._cache_dir / f"{digest}.json"
