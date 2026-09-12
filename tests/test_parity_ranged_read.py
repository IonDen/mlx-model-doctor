"""Tests for ``LocalTarget.read_bytes``'s ranged (offset + length) seek+read (F11).

The delta map (``parity.deltamap``) needs to byte-compare tensor payloads in
bounded chunks rather than reading a whole (potentially multi-gigabyte)
tensor into memory at once. These tests pin the new keyword-only
``offset``/``length`` seek+read against a real file on disk, and confirm the
pre-existing ``max_bytes`` (whole-file / prefix-read) behavior is unchanged
when ``length`` is omitted.
"""

from pathlib import Path

from mlx_model_doctor.targets import LocalTarget


def _write(tmp_path: Path, name: str, content: bytes) -> LocalTarget:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / name).write_bytes(content)
    return LocalTarget(model)


class TestRangedReadSlicesFromOffset:
    """``length`` given: seek to ``offset``, read exactly ``length`` bytes."""

    def test_reads_a_middle_slice_by_offset_and_length(self, tmp_path: Path) -> None:
        target = _write(tmp_path, "blob.bin", b"0123456789")
        # Byte 3 (10 * chunk_length[..3]) requires an actual seek, not a
        # from-the-top prefix read: a regression that ignored `offset` and
        # instead read `length` bytes from the start would return b"345"
        # rather than the true middle slice starting at 3.
        assert target.read_bytes("blob.bin", offset=3, length=4) == b"3456"

    def test_reads_from_zero_offset_with_length_matches_max_bytes_prefix(
        self, tmp_path: Path
    ) -> None:
        target = _write(tmp_path, "blob.bin", b"abcdefgh")
        assert target.read_bytes("blob.bin", offset=0, length=3) == b"abc"

    def test_length_past_eof_returns_only_the_remaining_bytes(self, tmp_path: Path) -> None:
        target = _write(tmp_path, "blob.bin", b"short")
        # A naive fixed-size read that didn't tolerate a short final chunk
        # would hang or raise instead of returning the tail.
        assert target.read_bytes("blob.bin", offset=2, length=100) == b"ort"

    def test_offset_at_exact_eof_returns_empty_bytes(self, tmp_path: Path) -> None:
        target = _write(tmp_path, "blob.bin", b"abc")
        assert target.read_bytes("blob.bin", offset=3, length=5) == b""


class TestOmittedLengthPreservesExistingBehavior:
    """``length=None`` (the default) must behave exactly as before this change."""

    def test_no_length_and_no_max_bytes_reads_the_whole_file(self, tmp_path: Path) -> None:
        target = _write(tmp_path, "blob.bin", b"whole-file-contents")
        assert target.read_bytes("blob.bin") == b"whole-file-contents"

    def test_no_length_with_max_bytes_still_reads_a_prefix(self, tmp_path: Path) -> None:
        target = _write(tmp_path, "blob.bin", b"abcdefgh")
        assert target.read_bytes("blob.bin", max_bytes=3) == b"abc"

    def test_offset_is_ignored_when_length_is_none(self, tmp_path: Path) -> None:
        # A regression that always seeked to `offset` (even without `length`)
        # would silently break every existing whole-file/max_bytes caller.
        target = _write(tmp_path, "blob.bin", b"abcdefgh")
        assert target.read_bytes("blob.bin", offset=5, max_bytes=3) == b"abc"
