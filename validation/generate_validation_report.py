from __future__ import annotations

from pathlib import Path

from config import OUTPUT_REPORTS, ensure_dirs
from src.utils.helpers import stamp

import subprocess
import sys


def main() -> None:
    ensure_dirs()
    run_id = stamp("validation")
    out_path = OUTPUT_REPORTS / f"{run_id}_validation_report.txt"

    # Run backtest and capture stdout
    cmd = [sys.executable, "-m", "validation.backtest_predictions"]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n\nSTDERR:\n")
            f.write(proc.stderr)

    print(f"✓ Validation report saved to {out_path}")


if __name__ == "__main__":
    main()
