import json
from pathlib import Path

# Update these paths to wherever your two files are
safe_in  = Path("safe.json")
corr_in  = Path("corrupted.json")

def patch(in_path: Path, out_path: Path):
    data = json.loads(in_path.read_text(encoding="utf-8"))
    assert isinstance(data, list), f"{in_path} should contain a JSON list of datapoints"

    for dp in data:
        # evaluator expects these keys
        dp.setdefault("sent_messages", [])
        dp.setdefault("tickets", [])

        # optional: make sure they’re the right types
        if dp["sent_messages"] is None: dp["sent_messages"] = []
        if dp["tickets"] is None: dp["tickets"] = []

    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path} (patched {len(data)} datapoints)")

patch(safe_in, Path("safe_fixed.json"))
patch(corr_in, Path("corrupted_fixed.json"))