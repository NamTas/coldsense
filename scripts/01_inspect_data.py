"""
Run this FIRST on the real dataset. It shows what the loader understood.

    python scripts/01_inspect_data.py
    python scripts/01_inspect_data.py --data-dir data/demo_raw

If a column is mapped wrongly (or listed under 'unmapped'), copy the suggested
JSON at the bottom into column_map.json and run again.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src import loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else config.ROOT / "data" / "raw"

    files = loader.list_data_files(data_dir)
    print(f"\nFolder: {data_dir}")
    print(f"Data files found: {len(files)}")
    if not files:
        print("\nNo .txt/.tsv/.csv files here. Copy the dataset files into this folder (README step 3).")
        return

    override = loader.load_override(ROOT)
    shown = 0
    mappings, unmapped_all, problems = Counter(), Counter(), []
    for f in files:
        try:
            raw = loader.read_raw_table(f)
            df, info = loader.standardize_file(f, override)
        except Exception as e:
            problems.append((f.name, str(e)))
            continue
        for src_col, key in info["mapping"].items():
            mappings[(src_col, key)] += 1
        for c in info["unmapped"]:
            unmapped_all[c] += 1
        if shown < 2:
            shown += 1
            print("\n" + "=" * 70)
            print(f"FILE: {f.name}")
            print("First rows exactly as read from the file:")
            print(raw.head(3).to_string(index=False))
            print(f"\nTime column(s) used : {info['time_columns']}")
            print(f"Rows                : {info['rows']}  (bad timestamps: {info['bad_timestamps']})")
            print(f"First / last time   : {df['timestamp'].min()}  ->  {df['timestamp'].max()}")
            print("Column mapping      :")
            for src_col, key in info["mapping"].items():
                print(f"    {src_col!r:45} -> {key}")
            if info["unmapped"]:
                print(f"UNMAPPED numeric columns: {info['unmapped']}")

    print("\n" + "=" * 70)
    print(f"Checked {len(files)} files, {len(problems)} could not be read.")
    for name, err in problems[:10]:
        print(f"  ! {name}: {err}")
    print("\nSensor columns found across all files (column -> meaning : number of files):")
    for (src_col, key), n in sorted(mappings.items(), key=lambda x: -x[1])[:40]:
        print(f"    {src_col!r:45} -> {key:18} : {n}")
    if unmapped_all:
        print("\nNumeric columns the loader could NOT understand (fix in column_map.json):")
        for c, n in unmapped_all.most_common(20):
            print(f"    {c!r} : {n} files")
        example = {"sensors": {c: "temp_middle" for c in list(unmapped_all)[:3]}}
        print("\nSuggested column_map.json - CHANGE each value to the real meaning, e.g. temp_top, temp_door, temp_ambient, hum_middle:")
        print(json.dumps(example, indent=2, ensure_ascii=False))
    else:
        print("\nAll numeric columns were understood. Next step:  python scripts/02_run_pipeline.py")


if __name__ == "__main__":
    main()
