"""Points every test at an isolated, throwaway data/ directory instead of
the project's real one. ``utils.paths`` resolves ``DATA_DIR`` from
``OPENFRONT_DATA_DIR`` at import time, so this must run before any test
module imports it — pytest imports ``conftest.py`` before collecting the
test modules in its directory tree, so top-level code here runs early
enough.
"""

import os
import tempfile

os.environ["OPENFRONT_DATA_DIR"] = tempfile.mkdtemp(prefix="openfront-il-test-")
