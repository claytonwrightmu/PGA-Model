from pathlib import Path
import pandas as pd
import re

RAW = Path("data/historical/raw/2026_uploaded_results_raw.tsv")
OUT = Path("data/historical/2026_results.csv")

STATUS_RE = re.compile(r"^(CUT|WD|W/D|DQ|DNS|MDF)$", re.IGNORECASE)
TIE_RE = re.compile(r"^T\s*(\d+)$", re.IGNORECASE)

def parse_finish(x):
    s = str(x).strip() if pd.notna(x) else ""
    if not s:
        return (float("nan"), "UNK")
    if STATUS_RE.match(s):
        return (float("nan"), STATUS_RE.match(s).group(1).upper().replace("/", ""))
    m = TIE_RE.match(s)
    if m:
        return (float(m.group(1)), "T")
    if s.isdigit():
        return (float(s), "FIN")
    return (float("nan"), "UNK")

def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Missing {RAW}")
    try:
        df = pd.read_csv(RAW, sep="\t")
        if "event_id" not in df.columns:
            df = pd.read_csv(RAW)
    except Exception:
        df = pd.read_csv(RAW)

    required = ["event_id","event_date","course_name","PLAYER_ID","PLAYER","finish_position","made_cut","winner"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    parsed = df["finish_position"].apply(parse_finish)
    df["finish_position_num"] = parsed.apply(lambda t: t[0])
    df["finish_status"] = parsed.apply(lambda t: t[1])
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.date
    df["made_cut"] = pd.to_numeric(df["made_cut"], errors="coerce").fillna(0).astype(int)
    df["winner"] = pd.to_numeric(df["winner"], errors="coerce").fillna(0).astype(int)
    df = df.drop_duplicates(subset=["event_id","PLAYER_ID","finish_position","made_cut","winner"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Built {OUT} | rows={len(df)} | events={df['event_id'].nunique()}")

if __name__ == "__main__":
    main()
