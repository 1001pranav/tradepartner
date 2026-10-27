#!/bin/sh
# Nightly backup (spec 14): SQLite online backup + config, kept 30 days.
# Usage: ops/backup.sh [HOME_DIR]   (default ~/.tradepartner)
set -eu
HOME_DIR="${1:-$HOME/.tradepartner}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
DEST="$HOME_DIR/backups/$STAMP"
mkdir -p "$DEST"
if [ -f "$HOME_DIR/db.sqlite" ]; then
  sqlite3 "$HOME_DIR/db.sqlite" ".backup '$DEST/db.sqlite'"
fi
tar -czf "$DEST/config.tgz" -C "$REPO_DIR" config
# keep 30 days
find "$HOME_DIR/backups" -mindepth 1 -maxdepth 1 -type d -mtime +30 -exec rm -rf {} +
echo "backup written to $DEST"
