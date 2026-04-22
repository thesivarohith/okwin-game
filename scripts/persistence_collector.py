import requests
import json
import time
import os
import csv
import pandas as pd
from datetime import datetime

# CONFIG
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "okwin_30s_dataset.csv")

def get_latest_rounds(page_no=1, retries=3):
    url = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    for attempt in range(retries):
        try:
            # Note: Public API might ignore page_no, but we send it just in case
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                return response.json().get("data", {}).get("list", [])
            else:
                print(f"API Error ({response.status_code}) on attempt {attempt+1}")
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
        
        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1)) # Exponential backoff
    return []

def run_collector(one_shot=False):
    print("══ OkWin Persistent Collector v3 ══")
    print(f"Mode: {'One-Shot' if one_shot else 'Looping'}")
    
    # Load existing periods
    existing_periods = set()
    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_csv(DATASET_PATH)
            existing_periods = set(df['period'].astype(str).tolist())
        except: pass

    # In one-shot mode, we fetch until we hit existing data (max 1000 pages / 10k rounds)
    fetch_limit = 1000 if one_shot else 1 
    
    while True:
        new_entries = []
        for p in range(1, fetch_limit + 1):
            rounds = get_latest_rounds(page_no=p)
            if not rounds: break # API failure
            
            found_new = False
            for r in rounds:
                pid = str(r.get("issueNumber"))
                if pid not in existing_periods:
                    num = int(r.get("number"))
                    size = "Big" if num >= 5 else "Small"
                    colors = {0: "Red+Violet", 1: "Green", 2: "Red", 3: "Green", 4: "Red", 
                              5: "Green+Violet", 6: "Red", 7: "Green", 8: "Red", 9: "Green"}
                    color_val = colors.get(num, "Unknown")
                    new_entries.append([datetime.now().isoformat(), pid, num, size, color_val])
                    existing_periods.add(pid)
                    found_new = True
            
            print(f"Page {p}: Found {sum(1 for e in new_entries if e[1] in [str(r.get('issueNumber')) for r in rounds])} new rounds")
            
            if not found_new: break # Linking successful! 
        
        if new_entries:
            new_entries.sort(key=lambda x: x[1])
            with open(DATASET_PATH, 'a') as f:
                writer = csv.writer(f)
                for entry in new_entries:
                    writer.writerow(entry)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] +{len(new_entries)} New Rounds | Total: {len(existing_periods):,}")

        # Update health check
        with open(os.path.join(os.path.dirname(DATASET_PATH), "scraper_status.json"), "w") as f:
            json.dump({
                "last_run": datetime.now().isoformat(),
                "total_rows": len(existing_periods),
                "status": "active"
            }, f)

        if one_shot: break
        time.sleep(30)

if __name__ == "__main__":
    import sys
    run_collector(one_shot="--one-shot" in sys.argv)
