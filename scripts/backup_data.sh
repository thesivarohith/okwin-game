#!/bin/bash

# OkWin Data Safety Backup
# Run this once to setup a cron job or run manually

DATA_DIR="/home/siva/Desktop/betique"
BACKUP_DIR="/home/siva/Desktop/betique/data/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

if [ -f "$DATA_DIR/okwin_30s_dataset.csv" ]; then
    cp "$DATA_DIR/okwin_30s_dataset.csv" "$BACKUP_DIR/okwin_30s_dataset_backup_$TIMESTAMP.csv"
    echo "Backup created: okwin_30s_dataset_backup_$TIMESTAMP.csv"
    
    # Keep only last 10 backups to save space
    ls -t "$BACKUP_DIR"/*.csv | tail -n +11 | xargs -r rm
else
    echo "Error: Dataset file not found at $DATA_DIR/okwin_30s_dataset.csv"
fi
