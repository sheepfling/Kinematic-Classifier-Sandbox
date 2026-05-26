from __future__ import annotations

import os
import tempfile
import unittest

from kinematic_classifier_sandbox.utils.runtime import (
    configure_matplotlib_environment,
    configure_runtime_environment,
    mpl_config_dir,
    pycache_prefix,
    runtime_root,
)


class UtilsRuntimeTests(unittest.TestCase):
    def test_runtime_paths_follow_override(self) -> None:
        original = os.environ.get("KINEMATIC_CLASSIFIER_RUNTIME_ROOT")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                os.environ["KINEMATIC_CLASSIFIER_RUNTIME_ROOT"] = temp_dir
                self.assertEqual(runtime_root(), type(runtime_root())(temp_dir))
                self.assertEqual(pycache_prefix(), type(runtime_root())(temp_dir) / "pycache")
                self.assertEqual(mpl_config_dir(), type(runtime_root())(temp_dir) / "mplconfig")
        finally:
            if original is None:
                os.environ.pop("KINEMATIC_CLASSIFIER_RUNTIME_ROOT", None)
            else:
                os.environ["KINEMATIC_CLASSIFIER_RUNTIME_ROOT"] = original

    def test_configure_runtime_environment_sets_defaults(self) -> None:
        original_pycache = os.environ.pop("PYTHONPYCACHEPREFIX", None)
        original_mpl = os.environ.pop("MPLCONFIGDIR", None)
        try:
            configure_runtime_environment()
            self.assertIn("PYTHONPYCACHEPREFIX", os.environ)
            self.assertIn("MPLCONFIGDIR", os.environ)
            os.environ.pop("PYTHONPYCACHEPREFIX", None)
            os.environ.pop("MPLCONFIGDIR", None)
            configure_matplotlib_environment()
            self.assertIn("MPLCONFIGDIR", os.environ)
        finally:
            if original_pycache is not None:
                os.environ["PYTHONPYCACHEPREFIX"] = original_pycache
            if original_mpl is not None:
                os.environ["MPLCONFIGDIR"] = original_mpl


if __name__ == "__main__":
    unittest.main()
