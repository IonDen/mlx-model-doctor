"""Assert `checks.__all__` matches the check classes the `text` plugin actually uses."""

import mlx_model_doctor.checks as checks_pkg
from mlx_model_doctor.plugins.text import TextModelPlugin


def test_checks_all_matches_exported_check_classes() -> None:
    exported = set(checks_pkg.__all__)
    # every name in __all__ is importable from the package
    for name in exported:
        assert hasattr(checks_pkg, name), name

    # every check class actually used by the text plugin (static_checks() /
    # weight_checks() / smoke_checks()) must be exported from __all__, so a new
    # check added to the plugin but forgotten in __all__ fails this test
    plugin = TextModelPlugin()
    used_checks = {
        type(check).__name__
        for check in (*plugin.static_checks(), *plugin.weight_checks(), *plugin.smoke_checks())
    }
    assert used_checks <= exported, used_checks - exported
