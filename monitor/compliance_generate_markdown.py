
# monitor/generate_report.py
from pathlib import Path
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import numpy as np

def nan_to_na(d: dict):
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = "N/A"
        elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out[k] = "N/A"
        else:
            out[k] = v
    return out

def main():
    # 取得 repo 根目錄與 monitor 目錄
    script_path = Path(__file__).resolve()
    monitor_dir = script_path.parent

    j2_path = monitor_dir / "compliance_report.md.j2"
    csv_path = monitor_dir / "metrics_log.csv"
    out_path = monitor_dir / "compliance_report.md"

    # 檔案存在性檢查（避免 FileNotFoundError）
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics_log.csv not found at: {csv_path}")
    if not j2_path.exists():
        raise FileNotFoundError(f"compliance_report.md.j2 not found at: {j2_path}")

    # 讀 metrics 最新一列
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError("metrics_log.csv is empty.")

    latest = df.tail(1).to_dict(orient="records")[0]
    latest = nan_to_na(latest)

    # 載入 Jinja2 模板（以 monitor 資料夾為根）
    env = Environment(loader=FileSystemLoader(str(monitor_dir)), autoescape=False)
    template = env.get_template("compliance_report.md.j2")

    # 多數模板長這樣：{{ run_name }}、{{ auc }}...
    rendered = template.render(**latest)

    # 輸出 Markdown
    out_path.write_text(rendered, encoding="utf-8")
    print(f"✅ Generated: {out_path}")

if __name__ == "__main__":
    main()
