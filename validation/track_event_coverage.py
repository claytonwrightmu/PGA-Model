from pathlib import Path
import pandas as pd
from datetime import date

SCHEDULE = Path("data/processed/tournaments/2026_event_schedule.csv")
RESULTS = Path("data/historical/2026_results.csv")

def main():
    if not SCHEDULE.exists():
        print(f"Missing schedule file: {SCHEDULE}")
        return

    sched = pd.read_csv(SCHEDULE)
    sched["end_date"] = pd.to_datetime(sched["end_date"], errors="coerce").dt.date

    uploaded = set()
    if RESULTS.exists():
        r = pd.read_csv(RESULTS)
        if "event_id" in r.columns:
            uploaded = set(r["event_id"].dropna().astype(str))

    completed = sched[sched["end_date"] <= date.today()].copy()
    completed["uploaded"] = completed["event_id"].isin(uploaded)
    missing = completed[~completed["uploaded"]]

    print("Completed:", len(completed))
    print("Uploaded:", int(completed["uploaded"].sum()))
    print("Missing:", len(missing))
    if len(missing):
        print("\nMissing event_ids:")
        for ev in missing["event_id"].tolist():
            print("-", ev)

if __name__ == "__main__":
    main()
