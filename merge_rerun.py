"""
merge_rerun.py

Move rerun rows from results_v2_<model>_<mode>_rerun/ into results_v2_<model>_<mode>/.
The row folder being replaced is moved to results_superseded/<same relative path>,
not deleted. Only rows with a finished index.json in the rerun folder are merged.

Usage:
    python merge_rerun.py            # preview what would be replaced
    python merge_rerun.py --apply    # do it
"""

import argparse
import shutil
from pathlib import Path

SUPERSEDED = Path("results_superseded")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    for rerun_root in sorted(Path(".").glob("results_v2_*_rerun")):
        target_root = Path(str(rerun_root)[: -len("_rerun")])
        for index in sorted(rerun_root.glob("env=*/model=*/adv=*/**/row=*/index.json")):
            new_row = index.parent
            rel = new_row.relative_to(rerun_root)
            old_row = target_root / rel
            backup = SUPERSEDED / target_root / rel
            action = "replace" if old_row.exists() else "add"
            print(f"[{action}] {old_row}  <-  {new_row}")
            if not args.apply:
                continue
            if old_row.exists():
                if backup.exists():
                    raise SystemExit(f"backup already exists, refusing to overwrite: {backup}")
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old_row), str(backup))
            old_row.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(new_row), str(old_row))

    if not args.apply:
        print("\nPreview only. Rerun with --apply to move the rows.")
