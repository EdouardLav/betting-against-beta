"""
Data loading utilities for the BAB replication project.

Sources:
- Kenneth French Data Library (stock returns, factors)
- AQR Capital Management (official BAB factors for validation)
"""

import io
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────
# Kenneth French Data Library
# ──────────────────────────────────────────────────────────

FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

FRENCH_DATASETS = {
    "ff3_monthly": "F-F_Research_Data_Factors",
    "ff5_monthly": "F-F_Research_Data_5_Factors_2x3",
    "momentum":    "F-F_Momentum_Factor",
    "10_portfolios_beta": "Portfolios_Formed_on_BETA",
    "25_size_beta": "25_Portfolios_ME_BETA_5x5",
}


def download_french_csv(dataset_name: str, cache_dir: str = "data") -> str:
    """Download a dataset from Ken French's library and return path to CSV."""
    import urllib.request

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    csv_path = cache_path / f"{dataset_name}.csv"

    if csv_path.exists():
        return str(csv_path)

    url = f"{FRENCH_BASE_URL}/{dataset_name}_CSV.zip"
    print(f"  Downloading {dataset_name} from French Library...")
    try:
        response = urllib.request.urlopen(url)
        zip_data = response.read()
    except Exception as e:
        raise ConnectionError(f"Failed to download {url}: {e}")

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        csv_name = [f for f in zf.namelist() if f.endswith('.CSV') or f.endswith('.csv')][0]
        with zf.open(csv_name) as f:
            content = f.read()
            csv_path.write_bytes(content)

    return str(csv_path)


def load_french_factors(dataset_key: str = "ff3_monthly",
                        cache_dir: str = "data") -> pd.DataFrame:
    """
    Load Fama-French factor data.
    Returns monthly returns in decimal (e.g., 0.01 = 1%).
    Index is DatetimeIndex at month-end.
    """
    name = FRENCH_DATASETS[dataset_key]
    csv_path = download_french_csv(name, cache_dir)

    with open(csv_path, 'r') as f:
        lines = f.readlines()

    # Find the start of monthly data (first line that starts with a 6-digit number YYYYMM)
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and stripped[0].isdigit() and len(stripped.split(',')[0].strip()) == 6:
            if start_idx is None:
                start_idx = i
        elif start_idx is not None and (not stripped or not stripped[0].isdigit() 
                                         or len(stripped.split(',')[0].strip()) != 6):
            end_idx = i
            break

    if end_idx is None:
        end_idx = len(lines)

    # Parse the header (line before start_idx)
    header_line = None
    for i in range(start_idx - 1, -1, -1):
        if lines[i].strip() and ',' in lines[i]:
            header_line = i
            break

    data_lines = [lines[header_line]] + lines[start_idx:end_idx]
    df = pd.read_csv(io.StringIO(''.join(data_lines)))

    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    first_col = df.columns[0]
    df = df.rename(columns={first_col: 'date_str'})
    df['date_str'] = df['date_str'].astype(str).str.strip()

    # Convert YYYYMM to datetime (month-end)
    df['date'] = pd.to_datetime(df['date_str'], format='%Y%m') + pd.offsets.MonthEnd(0)
    df = df.set_index('date').drop(columns=['date_str'])

    # Convert from percentage to decimal
    df = df.apply(pd.to_numeric, errors='coerce') / 100.0

    return df.sort_index()


def load_beta_sorted_portfolios(cache_dir: str = "data") -> dict:
    """
    Load the 10 beta-sorted portfolios from French Library.
    Returns dict with 'value_weighted' and 'equal_weighted' DataFrames.
    """
    name = FRENCH_DATASETS["10_portfolios_beta"]
    csv_path = download_french_csv(name, cache_dir)

    with open(csv_path, 'r') as f:
        content = f.read()

    # This file has multiple tables. We need to parse carefully.
    # Split by blank lines and find tables
    result = {}

    lines = content.split('\n')

    # Find sections
    sections = []
    current_section_name = None
    current_start = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if 'Value Weighted Returns' in stripped or 'Equal Weighted Returns' in stripped:
            if current_section_name:
                sections.append((current_section_name, current_start, i))
            current_section_name = 'value_weighted' if 'Value' in stripped else 'equal_weighted'
            current_start = i
    if current_section_name:
        sections.append((current_section_name, current_start, len(lines)))

    for sec_name, sec_start, sec_end in sections[:2]:
        sec_lines = lines[sec_start:sec_end]

        # Find header and data
        data_start = None
        header_idx = None
        for j, line in enumerate(sec_lines):
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and len(stripped.split()[0]) == 6:
                if data_start is None:
                    data_start = j
                    # header is one or two lines above
                    for k in range(j-1, max(0, j-3), -1):
                        if sec_lines[k].strip():
                            header_idx = k
                            break
            elif data_start is not None and (not stripped or not stripped[0].isdigit()):
                data_end = j
                break
        else:
            data_end = len(sec_lines)

        if data_start is None:
            continue

        block = [sec_lines[header_idx]] + sec_lines[data_start:data_end]
        df = pd.read_csv(io.StringIO('\n'.join(block)), sep=r'\s+')
        first_col = df.columns[0]
        df = df.rename(columns={first_col: 'date_str'})

        # Rename columns to P1..P10
        beta_cols = [c for c in df.columns if c != 'date_str']
        rename_map = {old: f'P{i+1}' for i, old in enumerate(beta_cols)}
        df = df.rename(columns=rename_map)

        df['date_str'] = df['date_str'].astype(str).str.strip()
        df['date'] = pd.to_datetime(df['date_str'], format='%Y%m') + pd.offsets.MonthEnd(0)
        df = df.set_index('date').drop(columns=['date_str'])
        df = df.apply(pd.to_numeric, errors='coerce') / 100.0

        result[sec_name] = df.sort_index()

    return result


# ──────────────────────────────────────────────────────────
# AQR Official BAB Data (for validation)
# ──────────────────────────────────────────────────────────

def load_aqr_data(filepath: str) -> dict:
    """
    Load AQR's official BAB factor data and supporting factors from their Excel file.
    Returns dict with keys: 'bab', 'mkt', 'smb', 'hml', 'umd', 'rf'
    """
    result = {}

    sheet_map = {
        'bab': 'BAB Factors',
        'mkt': 'MKT',
        'smb': 'SMB',
        'hml': 'HML FF',
        'umd': 'UMD',
        'rf':  'RF',
    }

    for key, sheet in sheet_map.items():
        try:
            df = pd.read_excel(filepath, sheet_name=sheet, skiprows=18, header=0)
        except Exception:
            print(f"  Warning: could not load sheet '{sheet}'")
            continue

        # Parse dates
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.set_index('DATE')
        df.index = df.index + pd.offsets.MonthEnd(0)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.sort_index()

        result[key] = df

    return result


# ──────────────────────────────────────────────────────────
# Convenience functions
# ──────────────────────────────────────────────────────────

def get_all_data(aqr_filepath: Optional[str] = None,
                 cache_dir: str = "data") -> dict:
    """
    Load all data needed for the replication.
    Returns dict with French factors, beta portfolios, and optionally AQR data.
    """
    print("Loading data...")

    data = {}

    # Fama-French 3 factors
    print("  [1/5] Fama-French 3 Factors")
    ff3 = load_french_factors("ff3_monthly", cache_dir)
    data['ff3'] = ff3

    # Fama-French 5 factors
    print("  [2/5] Fama-French 5 Factors")
    try:
        ff5 = load_french_factors("ff5_monthly", cache_dir)
        data['ff5'] = ff5
    except Exception:
        print("    Warning: FF5 not available")

    # Momentum factor
    print("  [3/5] Momentum Factor")
    try:
        mom = load_french_factors("momentum", cache_dir)
        data['mom'] = mom
    except Exception:
        print("    Warning: Momentum not available")

    # Beta-sorted portfolios
    print("  [4/5] Beta-sorted Portfolios")
    try:
        beta_portfolios = load_beta_sorted_portfolios(cache_dir)
        data['beta_portfolios'] = beta_portfolios
    except Exception as e:
        print(f"    Warning: Beta portfolios not available: {e}")

    # AQR data
    if aqr_filepath:
        print("  [5/5] AQR Official BAB Data")
        aqr = load_aqr_data(aqr_filepath)
        data['aqr'] = aqr

    print("Data loading complete.\n")
    return data
