#!/bin/bash
set -e

# Fix volume permissions (may have been created by previous root-running containers)
if [ "$(id -u)" = "0" ]; then
    chown -R appuser:appuser /app/indexed_pdfs 2>/dev/null || true
    chown -R appuser:appuser /home/appuser/.cache 2>/dev/null || true
    # Drop to non-root user and exec the CMD
    exec gosu appuser "$@"
else
    exec "$@"
fi
