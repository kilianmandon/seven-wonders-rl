#!/bin/bash

cd /root

curl -LsSf https://astral.sh/uv/install.sh | sh
source /root/.local/bin/env

mkdir alpha-zero-general
cd alpha-zero-general
uv venv --python 3.10
source .venv/bin/activate

uv pip install -r requirements.txt