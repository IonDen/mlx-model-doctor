from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_action_yml_is_a_composite_action_running_github_format() -> None:
    text = (REPO_ROOT / "action.yml").read_text(encoding="utf-8")

    assert 'using: "composite"' in text
    assert "--format github" in text
    assert "actions/setup-python@v6" in text
    # every documented output must be wired to the inner run step
    for output in ("pass", "warn", "fail", "skip", "exit-code", "schema-version"):
        assert f"steps.run.outputs.{output}" in text
    # the install step must be skippable so the self-test can use the local build
    assert "inputs.install == 'true'" in text
    # FIX A: injection-safe pattern — inputs are passed via env:, not interpolated into run:
    # the env: mapping must be present
    assert "INPUT_TARGET: ${{ inputs.target }}" in text
    # the run: script must consume the env var, not the raw expression
    assert '"$INPUT_TARGET"' in text
    # the old directly-interpolated forms (as they appeared in the run: script) must be gone
    assert '"${{ inputs.target }}"' not in text
    assert '"${{ inputs.source }}"' not in text


def test_pre_commit_hook_declares_check_local_entry() -> None:
    text = (REPO_ROOT / ".pre-commit-hooks.yaml").read_text(encoding="utf-8")

    assert "id: mlx-model-doctor" in text
    assert "entry: mlx-model-doctor check local" in text
    assert "language: python" in text
    assert "pass_filenames: false" in text
