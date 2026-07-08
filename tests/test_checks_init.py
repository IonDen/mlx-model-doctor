"""Assert `checks.__all__` matches check classes used by built-in plugins."""

import mlx_model_doctor.checks as checks_pkg
from mlx_model_doctor.plugins import BUILTIN_PLUGINS


def test_checks_all_matches_exported_check_classes() -> None:
    exported = set(checks_pkg.__all__)
    for name in exported:
        assert hasattr(checks_pkg, name), name

    used_checks = {
        type(check).__name__
        for plugin in BUILTIN_PLUGINS.values()
        for check in (*plugin.static_checks(), *plugin.weight_checks(), *plugin.smoke_checks())
    }
    assert used_checks <= exported, used_checks - exported
