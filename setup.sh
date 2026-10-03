#!/bin/bash
# Copyright 2024 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

# Prepare the Python model environment.

set -euo pipefail
script_dir="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly script_dir
cd -- "$script_dir"

if [ ! -d venv ]; then
	python3 -m venv venv
fi

"$script_dir/venv/bin/python" -m pip install -U pip
"$script_dir/venv/bin/python" -m pip install -r "$script_dir/requirements.txt"
