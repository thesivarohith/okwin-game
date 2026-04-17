import requests
import json
import time

def bulk_scrape():
    # REMINDER: Get your Bearer token from the browser Console using the Token Helper script.
    token = "YOUR_BEARER_TOKEN_HERE"
    
    base_url = "https://api.ar-lottery01.com/api/Lottery/GetHistoryIssuePage"
    
    # These params were captured from a real session. 
    # Update them with fresh ones from your Network tab if they expire.
    params = {
        "gameCode": "WinGo_30S",
        "pageSize": 10,
        "language": "en",
        "random": "782885479074",
        "signature": "977CCD71D6D72B69F67A0D93C54A2350",
        "timestamp": "1776406091"
    }
    
    headers = {
        "Authorization": token,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://2okwin6.com",
        "Referer": "https://2okwin6.com/"
    }
    
    results = []
    
    print("Starting mass scrape...")
    for i in range(1, 1500):
        params["pageNo"] = i
        print(f"Fetching page {i}...", end="\r")
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                items = data.get("data", {}).get("list", [])
                if not items:
                    print(f"\nStopped at page {i}: No more items.")
                    break
                results.extend(items)
            else:
                print(f"\nError {response.status_code} on page {i}")
                break
        except Exception as e:
            print(f"\nException: {e}")
            break
        time.sleep(0.3)
    
    if results:
        with open("data/scraped_history.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} items to data/scraped_history.json")

if __name__ == "__main__":
    bulk_scrape()
