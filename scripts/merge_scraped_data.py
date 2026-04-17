import json
import pandas as pd
import os

def merge_data():
    # Looks for any JSON files in root or data/
    json_files = [f for f in os.listdir(".") if f.endswith(".json")]
    json_files += [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".json")]
    
    unique_data = {}
    
    for file_path in json_files:
        if not os.path.exists(file_path) or "package" in file_path:
            continue
            
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list): continue
                for item in data:
                    period = item.get("period") or item.get("issueNumber")
                    number = item.get("number")
                    if period and number is not None:
                        num = int(number)
                        size = "Big" if num >= 5 else "Small"
                        
                        # Color logic
                        if num == 0: color = "Red+Violet"
                        elif num == 5: color = "Green+Violet"
                        elif num in [1, 3, 7, 9]: color = "Green"
                        else: color = "Red"
                        
                        unique_data[str(period)] = {
                            "timestamp": str(period)[:10],
                            "period": str(period),
                            "result": int(number),
                            "size": size,
                            "color": color
                        }
            except: continue
    
    # Load existing CSV
    csv_path = "okwin_30s_dataset.csv"
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        existing_df["period"] = existing_df["period"].astype(str)
        new_df = pd.DataFrame(unique_data.values())
        final_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=["period"], keep="last")
        final_df = final_df.sort_values(by="period", ascending=False)
        final_df.to_csv("data/raw_data.csv", index=False)
        print(f"Merged successfully. Total unique records: {len(final_df)}")

if __name__ == "__main__":
    merge_data()
