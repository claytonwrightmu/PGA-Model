from pathlib import Path
import argparse
import re
import pandas as pd

STATUS_RE = re.compile(r"^(CUT|WD|W/D|DQ|DNS|MDF)$", re.IGNORECASE)
TIE_RE = re.compile(r"^T\s*(\d+)$", re.IGNORECASE)

def parse_finish(x):
    s = str(x).strip() if pd.notna(x) else ""
    if not s:
        return (float("nan"), "UNK")
    m = STATUS_RE.match(s)
    if m:
        return (float("nan"), m.group(1).upper().replace("/", ""))
    t = TIE_RE.match(s)
    if t:
        return (float(t.group(1)), "T")
    if s.isdigit():
        return (float(s), "FIN")
    return (float("nan"), "UNK")

ap = argparse.ArgumentParser()
ap.add_argument("--year", required=True, type=int)
args = ap.parse_args()

raw = Path(f"data/historical/raw/{args.year}_uploaded_results_raw.tsv")
out = Path(f"data/historical/{args.year}_results.csv")

if not raw.exists():
    raise FileNotFoundError(f"Missing raw file: {raw}")

try:
    df = pd.read_csv(raw, sep="\t")
    if "event_id" not in df.columns:
        df = pd.read_csv(raw)
except Exception:
    df = pd.read_csv(raw)

required = ["event_id","event_date","course_name","PLAYER_ID","PLAYER","finish_position","made_cut","winner"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"{raw} missing columns: {missing}")

# drop accidental duplicated header rows
df = df[df["event_id"].astype(str).str.lower() != "event_id"]

parsed = df["finish_position"].apply(parse_finish)
df["finish_position_num"] = parsed.apply(lambda t: t[0])
df["finish_status"] = parsed.apply(lambda t: t[1])

df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.date
df["made_cut"] = pd.to_numeric(df["made_cut"], errors="coerce").fillna(0).astype(int)
df["winner"] = pd.to_numeric(df["winner"], errors="coerce").fillna(0).astype(int)

df = df.drop_duplicates(subset=["event_id","PLAYER_ID","finish_position","made_cut","winner"]).reset_index(drop=True)
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print(f"✓ Built {out} | rows={len(df)} | events={df['event_id'].nunique()}")
