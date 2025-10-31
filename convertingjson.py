import pandas as pd
import json
from pathlib import Path

# Input Excel file name
INPUT_XLSX = "SCAQ Collated.xlsx"
OUT_DIR = Path("excel_sheets_json")
OUT_DIR.mkdir(exist_ok=True)

def _dedupe_headers(cols):
    """Clean and deduplicate Excel column headers."""
    seen = {}
    new_cols = []
    for c in cols:
        base = str(c).strip() if c is not None else ""
        if base == "":
            base = "Unnamed"
        if base not in seen:
            seen[base] = 1
            new_cols.append(base)
        else:
            seen[base] += 1
            new_cols.append(f"{base}_{seen[base]}")
    return new_cols

# Read all sheets from the Excel file
xls = pd.read_excel(INPUT_XLSX, sheet_name=None, dtype=str)

for sheet_name, df in xls.items():
    print(f"\n=== Sheet: {sheet_name} ===")
    df.columns = _dedupe_headers(df.columns)
    df = df.loc[:, ~df.isna().all(axis=0)]  # drop columns that are entirely empty

    records = []
    for i, row in df.iterrows():
        record = {}
        print(f"\nRow {i+1}:")
        for col in df.columns:
            val = str(row[col]).strip() if pd.notna(row[col]) else None
            record[col] = val
            print(f"  {col}: {val}")  # <-- This prints all key-value pairs
        records.append(record)

    # Write JSON file for this sheet
    out_path = OUT_DIR / f"{sheet_name}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Wrote {len(records)} rows to {out_path}")
