"""Tests for the HF listing cache."""

import time
from pathlib import Path

from mlx_model_doctor.cache import CacheKey, ListingCache


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = ListingCache(cache_dir=tmp_path, ttl_seconds=3600)
    assert cache.get(("mlx-community", None, 200)) is None


def test_cache_put_then_get(tmp_path: Path) -> None:
    cache = ListingCache(cache_dir=tmp_path, ttl_seconds=3600)
    key: CacheKey = ("mlx-community", None, 200)
    data = [{"repo_id": "mlx-community/Llama", "tags": ["mlx"], "library_name": "mlx"}]
    cache.put(key, data)
    result = cache.get(key)
    assert result == data


def test_cache_expires_after_ttl(tmp_path: Path) -> None:
    cache = ListingCache(cache_dir=tmp_path, ttl_seconds=0)
    key: CacheKey = ("mlx-community", None, 200)
    cache.put(key, [{"repo_id": "a", "tags": [], "library_name": None}])
    time.sleep(0.01)
    assert cache.get(key) is None


def test_cache_different_keys_independent(tmp_path: Path) -> None:
    cache = ListingCache(cache_dir=tmp_path, ttl_seconds=3600)
    k1: CacheKey = ("mlx-community", None, 200)
    k2: CacheKey = ("mlx-community", "text-generation", 200)
    cache.put(k1, [{"repo_id": "a", "tags": [], "library_name": None}])
    assert cache.get(k2) is None


def test_cache_clear(tmp_path: Path) -> None:
    cache = ListingCache(cache_dir=tmp_path, ttl_seconds=3600)
    key: CacheKey = ("mlx-community", None, 200)
    cache.put(key, [{"repo_id": "a", "tags": [], "library_name": None}])
    cache.clear()
    assert cache.get(key) is None


def test_cache_survives_corrupt_file(tmp_path: Path) -> None:
    cache = ListingCache(cache_dir=tmp_path, ttl_seconds=3600)
    key: CacheKey = ("mlx-community", None, 200)
    cache_file = cache._path_for(key)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("not json", encoding="utf-8")
    assert cache.get(key) is None


def test_cache_none_task_and_string_none_have_different_keys(tmp_path: Path) -> None:
    cache = ListingCache(cache_dir=tmp_path, ttl_seconds=3600)
    data_none: list[dict[str, object]] = [{"repo_id": "a", "tags": [], "library_name": None}]
    data_str: list[dict[str, object]] = [{"repo_id": "b", "tags": [], "library_name": None}]
    cache.put(("author", None, 200), data_none)
    cache.put(("author", "None", 200), data_str)
    assert cache.get(("author", None, 200)) == data_none
    assert cache.get(("author", "None", 200)) == data_str
