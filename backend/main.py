"""
FastAPI backend for OkWin Big/Small Predictor.
Endpoints: POST /predict, GET /health, GET /stats, GET /auto-predict, GET /training_status
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
        
    try:
        with open(DATASET_PATH, 'r') as f:
            lines = f.readlines()[-50:]
            if any(period_str in l for l in lines):
                return
    except:
        pass
        
    ts = datetime.now().isoformat()
    size = "Big" if number >= 5 else "Small"
    colors = {0: "Red+Violet", 1: "Green", 2: "Red", 3: "Green", 4: "Red", 
              5: "Green+Violet", 6: "Red", 7: "Green", 8: "Red", 9: "Green"}
    color_val = colors.get(number, "Unknown")
        
    with open(DATASET_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ts, period_str, number, size, color_val])

sys.path.insert(0, os.path.dirname(__file__))
from predict import PredictionEngine

app = FastAPI(
    title="OkWin Big/Small Predictor API v2",
    version="2.0.0",
    description="AI-powered ensemble prediction for OkWin Big/Small outcomes"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = PredictionEngine()
engine.load_models() # Safe, turns True if all exist

prediction_history = []
MAX_HISTORY = 100
_okwin_buffer = {}

class PredictRequest(BaseModel):
    history: List[int]
    period: str

class PredictResponse(BaseModel):
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[dict] = None
    model_votes: Optional[dict] = None
    kalman_drift: Optional[float] = None
    cycle_phase: Optional[float] = None
    timestamp: str
    target_period: str
    history: Optional[List[int]] = None
    status: Optional[str] = "success"
    message: Optional[str] = ""

@app.post("/predict")
async def predict(request: PredictRequest):
    global prediction_history
    if len(request.history) < 20: # frontend might send 20, we auto pad
        return {"status": "error", "message": "Need more history"}

    # predict.py now expects 50 length history. If frontend sends 20, pad with 0
    hist = request.history
    while len(hist) < 50:
        hist.insert(0, 0)

    try:
        period_int = int(request.period)
        result = engine.predict(hist, period_int)
        
        if "status" in result and result["status"] != "success":
            raise HTTPException(status_code=500, detail=result.get("message", "Models not ready"))
            
        result['timestamp'] = time.time()
        result['target_period'] = str(request.period)
        result['history'] = request.history
        
        global prediction_history
        prediction_history = [p for p in prediction_history if p.get("target_period") != result['target_period']]
        prediction_history.append(result)
        if len(prediction_history) > MAX_HISTORY:
            prediction_history.pop(0)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy" if engine.loaded else "not_ready",
        "version": "2.0.0",
        "models_loaded": engine.loaded,
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/stats")
async def stats():
    total = len(prediction_history)
    if total == 0:
        return {"total_predictions": 0, "success_rate": 0, "recent_predictions": []}

    graded = [p for p in prediction_history if "correct" in p]
    success_rate = 0.0
    if graded:
        wins = sum(1 for p in graded if p["correct"])
        success_rate = round((wins / len(graded)) * 100, 1)

    return {
        "total_predictions": total,
        "graded_predictions": len(graded),
        "success_rate": success_rate,
        "recent_predictions": prediction_history[-50:][::-1],
    }

@app.get("/auto-predict")
async def auto_predict():
    global prediction_history, _okwin_buffer
    try:
        base_url = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        r1 = requests.get(base_url, headers=headers, timeout=5)
        items = r1.json().get("data", {}).get("list", [])
        
        for item in items:
            if item["issueNumber"] not in _okwin_buffer:
                _okwin_buffer[item["issueNumber"]] = int(item["number"])
        
        sorted_issues = sorted(_okwin_buffer.keys(), key=int)
        if len(sorted_issues) > 100:
            for old_key in sorted_issues[:-100]:
                del _okwin_buffer[old_key]
            sorted_issues = sorted(_okwin_buffer.keys(), key=int)
        
        last_50_issues = sorted_issues[-50:]
        history_for_model = [_okwin_buffer[k] for k in last_50_issues]
        while len(history_for_model) < 50:
            history_for_model.insert(0, 0)
        
        for p in prediction_history:
            tp = p.get("target_period")
            if tp and "correct" not in p:
                tp_str = str(tp)
                if tp_str in _okwin_buffer:
                    actual = _okwin_buffer[tp_str]
                    actual_label = "Big" if actual >= 5 else "Small"
                    p["correct"] = (p.get("prediction") == actual_label)
                    log_to_dataset(tp_str, actual)
        
        state_r = requests.get("https://draw.ar-lottery01.com/WinGo/WinGo_30S.json", headers=headers, timeout=5)
        state_data = state_r.json()
        target_period_str = str(state_data.get("current", {}).get("issueNumber", ""))
        
        result = engine.predict(history_for_model, int(target_period_str))
        
        if "status" in result and result.get("status") != "success":
            raise HTTPException(status_code=500, detail=result.get("message", "Models not ready"))
            
        result['timestamp'] = time.time()
        # Frontend UI only wants 20 historical items to render visually
        result['history'] = history_for_model[-20:]
        result['target_period'] = target_period_str
        
        prediction_history = [p for p in prediction_history if p.get("target_period") != target_period_str]
        prediction_history.append(result)
        if len(prediction_history) > MAX_HISTORY:
            prediction_history.pop(0)
            
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training_status")
async def training_status():
    report_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'eval_report.txt')
    data_path = DATASET_PATH
    
    rows = 0
    if os.path.exists(data_path):
        import pandas as pd
        rows = len(pd.read_csv(data_path))
        
    last_accuracy = "not trained yet"
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "FINAL STACKED ACCURACY" in line:
                    last_accuracy = line.split(":")[-1].strip()
                    break
                    
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'model', 'saved')
    found, missing = [], []
    req_models = {
        "xgb": "model_xgb.pkl",
        "lstm": "model_lstm.keras",
        "tcn": "model_tcn.keras",
        "hmm": "model_hmm.pkl",
        "markov": "markov_model.pkl",
        "meta_mlp": "model_meta_mlp.keras"
    }
    
    for name, f in req_models.items():
        if os.path.exists(os.path.join(models_dir, f)):
            found.append(name)
        else:
            missing.append(name)
            
    return {
        "models_ready": len(missing) == 0,
        "models_found": found,
        "models_missing": missing,
        "real_data_rows": rows,
        "last_accuracy": last_accuracy
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
