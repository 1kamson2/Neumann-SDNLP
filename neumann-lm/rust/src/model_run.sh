#!/bin/bash
readonly ROOT="/home/kums0n-desktop/Dev/lm-project/neumann-lm"
readonly SOURCE_FILE="$ROOT/python/.venv/bin/activate"
readonly MAIN_FILE="$ROOT/python/main.py"
run_model() {

  if [[ -f "$SOURCE_FILE" && -f "$MAIN_FILE" ]]; then
    source "$SOURCE_FILE" || exit
    python3 "$MAIN_FILE" "$@"
  else
    printf "The script couldn't run with the following arguments:\n"
    echo "$@"
  fi
  return
}

run_model "$@"
