#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

if python -c 'import mujoco' >/dev/null 2>&1; then
  exec python "$script_dir/visualize.py" "$@"
fi

existing_python="/home/dsa/project/xivirobotpi/mainproj/openpi/.venv/bin/python"
if [ -x "$existing_python" ] && "$existing_python" -c 'import mujoco' >/dev/null 2>&1; then
  exec "$existing_python" "$script_dir/visualize.py" "$@"
fi

printf '%s\n' \
  "MuJoCo is not installed in the current Python environment." \
  "Install it with: python -m pip install mujoco" >&2
exit 1
