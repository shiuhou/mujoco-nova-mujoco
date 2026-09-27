"""Choose offscreen EGL before any test imports MuJoCo; deny hardware I/O."""

import os

from s4_offline_guard import install

os.environ.setdefault("MUJOCO_GL", "egl")
S4_BLOCKED_ATTEMPTS = install()
