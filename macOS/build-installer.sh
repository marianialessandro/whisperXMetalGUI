#!/bin/sh
set -eu
app_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec uv run --no-project --python 3.12.12 "$app_root/Packaging/build_installer.py" "$@"
