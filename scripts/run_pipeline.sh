#!/bin/bash
echo "🚀 Starting OkWin AI Update Pipeline..."

# 1. Merge all JSON scrapes
python3 scripts/merge_scraped_data.py

# 2. Augment to 15,000 samples
python3 scripts/augment_data.py

# 3. Preprocess and Feature Engineering
python3 training/preprocess.py
python3 training/feature_engineering.py

# 4. Retrain Models
python3 training/train_xgboost.py
python3 training/train_lstm.py
python3 training/train_markov.py

# 5. Restart Backend
echo "Restarting backend..."
pkill -f "python3 main.py" || true
nohup python3 backend/main.py > backend/server.log 2>&1 &

echo "✅ Pipeline Complete! AI is now running on updated 15,000-sample brain."
