import requests
import json
import time
import os
import csv
import pandas as pd
from datetime import datetime

# CONFIG
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "okwin_30s_dataset.csv")
# CONFIG
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "okwin_30s_dataset.csv")
POLL_INTERVAL = 30  # Used only in loop mode

def get_latest_rounds(page_no=1):
    # This public endpoint NO LONGER requires a TOKEN!
    url = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json().get("data", {}).get("list", [])
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] API Error: {e}")
    return []

def run_collector(one_shot=False):
    print("══ OkWin Persistent Collector ══")
    print(f"Mode: {'One-Shot' if one_shot else 'Looping'}")
    print(f"Recording to: {DATASET_PATH}")
    
    # Load existing periods to avoid duplicates
    existing_periods = set()
    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_csv(DATASET_PATH)
            existing_periods = set(df['period'].astype(str).tolist())
        except:
            pass

    max_pages = 10 if one_shot else 1 # Fetch more if one-shot to catch up on 30m gap
    
    while True:
        new_entries = []
        for p in range(1, max_pages + 1):
            # We need to pass pageNo to get_latest_rounds
            # Let's quickly modify get_latest_rounds to accept pageNo
            rounds = get_latest_rounds(page_no=p)
            found_new_on_page = False
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
                    found_new_on_page = True
            
            if not found_new_on_page and p > 1: # If p1 has nothing new, p2 won't either
                break
        
        if new_entries:
            new_entries.sort(key=lambda x: x[1])
            with open(DATASET_PATH, 'a') as f:
                writer = csv.writer(f)
                for entry in new_entries:
                    writer.writerow(entry)
            
            total_count = len(existing_periods)
            percent = (total_count / 20000) * 100
            print(f"[{datetime.now().strftime('%H:%M:%S')}] +{len(new_entries)} Rows | Total: {total_count:,} ({percent:.1f}%)")

        if one_shot:
            break
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    import sys
    is_oneshot = "--one-shot" in sys.argv
    run_collector(one_shot=is_oneshot)
