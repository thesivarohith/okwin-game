"""
FastAPI backend for OkWin Big/Small Predictor.
Endpoints: POST /predict, GET /health, GET /stats
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
import sys
import requests
import csv
import time

DATASET_PATH = "/home/siva/Desktop/betique/okwin_30s_dataset.csv"

def log_to_dataset(period_str, number):
    """Append a new result to the CSV dataset if not already present."""
    if not os.path.exists(DATASET_PATH):
        return
        
    # Check if period already in file (last 50 lines)
    try:
        with open(DATASET_PATH, 'r') as f:
            lines = f.readlines()[-50:]
            if any(period_str in l for l in lines):
                return
    except:
        pass
        
    # Derive additional columns
    ts = time.time()
    size = "Big" if number >= 5 else "Small"
    # Color map
    colors = {0: "Red+Violet", 1: "Green", 2: "Red", 3: "Green", 4: "Red", 
              5: "Green+Violet", 6: "Red", 7: "Green", 8: "Red", 9: "Green"}
    color_val = colors.get(number, "Unknown")
        
    with open(DATASET_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ts, period_str, number, size, color_val])


# Add backend dir to path
sys.path.insert(0, os.path.dirname(__file__))
from predict import PredictionEngine

app = FastAPI(
    title="OkWin Big/Small Predictor API",
    version="1.0.0",
    description="AI-powered ensemble prediction for OkWin Big/Small outcomes"
)

# CORS — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global prediction engine
engine = PredictionEngine()

# Store recent predictions for stats
prediction_history = []
MAX_HISTORY = 100

# Rolling buffer for OkWin history: {issueNumber_str: result_number}
_okwin_buffer = {}


# --- Request/Response Models ---

class PredictRequest(BaseModel):
    history: List[int]  # Last 20+ result values (0-9)
    period: str         # Current period number (string to avoid precision loss)

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    model_contributions: dict
    timestamp: str
    target_period: str  # String for ID safety
    history: Optional[List[int]] = None


# --- Startup ---

# Load models once at startup (global scope for reliability)
try:
    print("Initializing PredictionEngine...")
    engine.load_models()
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FATAL: Failed to load models: {e}")
    sys.exit(1)


# --- Endpoints ---

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict Big or Small for the next round."""
    global prediction_history
    if len(request.history) < 20:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 20 history values, got {len(request.history)}"
        )

    # Validate values are 0-9
    for v in request.history:
        if not (0 <= v <= 9):
            raise HTTPException(
                status_code=400,
                detail=f"History values must be 0-9, got {v}"
            )

    try:
        # Predict uses int for logic, but we receive str for API safety/precision
        period_int = int(request.period)
        result = engine.predict(request.history, period_int)
        
        result['timestamp'] = time.time()
        result['target_period'] = str(request.period)
        result['history'] = request.history

        # Track history (overwrite if predicting same period again)
        prediction_history = [p for p in prediction_history if p.get("target_period") != result['target_period']]
        
        prediction_history.append(result)
        if len(prediction_history) > MAX_HISTORY:
            prediction_history.pop(0)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """API health check."""
    return {
        "status": "healthy" if engine.loaded else "not_ready",
        "version": "1.0.0",
        "models_loaded": engine.loaded,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/stats")
async def stats():
    """Prediction stats and history."""
    total = len(prediction_history)
    if total == 0:
        return {
            "total_predictions": 0,
            "big_predictions": 0,
            "small_predictions": 0,
            "avg_confidence": 0,
            "recent_predictions": [],
        }

    big_count = sum(1 for p in prediction_history if p['prediction'] == 'Big')
    small_count = total - big_count
    avg_conf = sum(p['confidence'] for p in prediction_history) / total

    graded = [p for p in prediction_history if "correct" in p]
    success_rate = 0.0
    if graded:
        wins = sum(1 for p in graded if p["correct"])
        success_rate = round((wins / len(graded)) * 100, 1)

    return {
        "total_predictions": total,
        "graded_predictions": len(graded),
        "success_rate": success_rate,
        "big_predictions": big_count,
        "small_predictions": small_count,
        "big_ratio": round(big_count / total * 100, 1),
        "avg_confidence": round(avg_conf, 2),
        "recent_predictions": prediction_history[-50:][::-1],
    }


@app.get("/auto-predict")
async def auto_predict():
    """Fetch live data from OkWin and auto-run the prediction."""
    global prediction_history, _okwin_buffer
    try:
        base_url = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json, text/plain, */*",
        }
        
        # Fetch latest 10 results from OkWin (API hard-caps at 10)
        r1 = requests.get(base_url, headers=headers, timeout=5)
        items = r1.json().get("data", {}).get("list", [])
        
        # Add to rolling buffer (keyed by issueNumber to prevent duplicates)
        new_count = 0
        for item in items:
            if item["issueNumber"] not in _okwin_buffer:
                _okwin_buffer[item["issueNumber"]] = int(item["number"])
                new_count += 1
        
        if new_count > 0:
            print(f"Sync: Added {new_count} new results to buffer.")
        
        # Sort buffer by issueNumber (ascending) and keep last 30
        sorted_issues = sorted(_okwin_buffer.keys(), key=int)
        if len(sorted_issues) > 30:
            for old_key in sorted_issues[:-30]:
                del _okwin_buffer[old_key]
            sorted_issues = sorted(_okwin_buffer.keys(), key=int)
        
        # Take last 20 results for the model
        last_20_issues = sorted_issues[-20:] if len(sorted_issues) >= 20 else sorted_issues
        history_for_model = [_okwin_buffer[k] for k in last_20_issues]
        
        # Pad with zeros if we don't have 20 yet (first ~5 minutes)
        while len(history_for_model) < 20:
            history_for_model.insert(0, 0)
        
        # Grade ALL past predictions against known results
        graded_count = 0
        for p in prediction_history:
            tp = p.get("target_period")
            if tp and "correct" not in p:
                tp_str = str(tp)
                if tp_str in _okwin_buffer:
                    actual = _okwin_buffer[tp_str]
                    actual_label = "Big" if actual >= 5 else "Small"
                    # New result confirmed!
                    p["correct"] = (p["prediction"] == actual_label)
                    p["actual_result"] = actual
                    graded_count += 1
                    # Persist to disk for retraining
                    log_to_dataset(tp_str, actual)
        
        if graded_count > 0:
            print(f"Grading: Result confirmed for {graded_count} predictions.")
        
        # Get current period
        state_r = requests.get("https://draw.ar-lottery01.com/WinGo/WinGo_30S.json", headers=headers, timeout=5)
        state_data = state_r.json()
        target_period_str = str(state_data.get("current", {}).get("issueNumber", ""))
        
        if not target_period_str:
            raise HTTPException(status_code=500, detail="Target period ID missing from OkWin API")

        # Run prediction
        result = engine.predict(history_for_model, int(target_period_str))
        result['timestamp'] = time.time()
        result['history'] = history_for_model
        result['target_period'] = target_period_str
        result['buffer_size'] = len(sorted_issues)
        
        # Track history (one per period)
        prediction_history = [p for p in prediction_history if p.get("target_period") != target_period_str]
        prediction_history.append(result)
        if len(prediction_history) > MAX_HISTORY:
            prediction_history.pop(0)
            
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
