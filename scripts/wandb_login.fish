#!/usr/bin/env fish
# This script never receives or stores an API key in the repository.
# wandb prompts for the key without echoing it and saves credentials outside
# this checkout.

set -l wandb_host "https://wandb-radfan.ru"
uv run --with wandb wandb login --host=$wandb_host --relogin
