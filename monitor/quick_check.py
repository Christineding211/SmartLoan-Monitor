
# quick_check.py
import pickle, yaml
rs = pickle.load(open("monitor/reference_stats.pkl","rb"))
cfg = yaml.safe_load(open("monitor/config.yaml"))
nums = cfg["features"]["numerical"]

for col in nums:
    s = rs.get("features", {}).get(col, {})
    print(f"{col:24} bins={ 'bins' in s }  counts={ 'counts' in s }")
