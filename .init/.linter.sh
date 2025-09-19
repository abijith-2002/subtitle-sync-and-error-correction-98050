#!/bin/bash
cd /home/kavia/workspace/code-generation/subtitle-sync-and-error-correction-98050/subtitle_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

