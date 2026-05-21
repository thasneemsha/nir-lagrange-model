"""
data_loader.py  –  CS334 Data Mining Project
=============================================
Shared utility: loads and cleans the NIR glucose dataset.
All four technique scripts import from here so we don't
repeat the loading code everywhere.

Dataset facts (from README):
  • Calibration : 273 samples
  • Validation  : 120 samples
  • 9 info columns  → Glucose, Lactate, Acetaminophen, Caffeine,
                       Ethanol, Temperature, Cuvette, Day, Run
  • 4200 wavelength columns  → 400 nm to 2499.5 nm (step 0.5 nm)
  • Files are UTF-16 LE encoded, tab-separated
"""

from pathlib import Path
import numpy as np  #converts data into fast numeric arrays
import pandas as pd #reads the .txt table

# Base project folder
BASE_DIR = Path(__file__).resolve().parent

# ── file paths ──────────────────────────────────────────────────────────────
CAL_PATH = BASE_DIR / 'data' / 'CalibrationData.txt'
VAL_PATH = BASE_DIR / 'data' / 'ValidationData.txt'

# ── column name constants ────────────────────────────────────────────────────
INFO_COLS = [
    'Glucose (mM)',
    'Lactate (mM)',
    'Acetaminophen (mM)',
    'Caffeine (mM)',
    'Ethanol (mM)',
    'Temperature (C)',
    'Cuvette',
    'Day',
    'Run',
]

# Glucose level bins used in Techniques 02, 03, 04
#   Low   : 0 – 10 mM   (hypoglycaemia range)
#   Normal: 10 – 30 mM  (healthy fasting range, roughly)
#   High  : 30 – 50 mM  (hyperglycaemia range)
GLUCOSE_BINS   = [0, 10, 30, 50]
GLUCOSE_LABELS = ['Low', 'Normal', 'High']


def load_raw(path, max_real_cols=4209):
    """
     - Reads the raw spectrometer file
     - Removes extra junk columns added by the instrument
     - Keeps only the real 4200 wavelength features

    Parameters
    ----------
    path          : str  – file path [raw file contains extra empty columns]
    max_real_cols : int  – number of columns to keep (default 4209)
    -> Only the first 4209 columns are actually useful data

    Returns
    -------
    df : pd.DataFrame with clean column names
    """

    df = pd.read_csv(path, encoding='utf-16', sep='\t', header=0)
    #utf-16 -> spectrometer exported weird encoding
    #sep='\t' -> tab-separated file (not CSV)
    # header=0 -> first row = column names

    # Ignore junk columns, keep only real data
    df = df.iloc[:, :max_real_cols]

    # Strip trailing whitespace, and weird format from column names
    df.columns = [c.strip() for c in df.columns]

    return df


def load_dataset():
    """
    Load, clean, and split the NIR dataset into:
      X_cal  – calibration spectra  (273 × 4200 float array)
      y_cal  – calibration glucose  (273,)
      X_val  – validation spectra   (120 × 4200 float array)
      y_val  – validation glucose   (120,)
      wavelengths  – 4200-element array of nm values
      info_cal / info_val  – DataFrames with all 9 info columns

    Returns
    -------
    A dict with all of the above.
    """
    print("Loading calibration data …")
    df_cal = load_raw(CAL_PATH)

    print("Loading validation data …")
    df_val = load_raw(VAL_PATH)

    # ── Rename info columns to clean names ──────────────────────────────────
    rename_map = {old: new for old, new in
                  zip(df_cal.columns[:9], INFO_COLS)}
    df_cal.rename(columns=rename_map, inplace=True)
    df_val.rename(columns=rename_map, inplace=True)

    # ── Separate info vs spectra ─────────────────────────────────────────────
    info_cal = df_cal[INFO_COLS].copy()
    info_val = df_val[INFO_COLS].copy()

    # Wavelength columns: everything after the 9 info cols
    wave_cols   = df_cal.columns[9:].tolist()
    wavelengths = np.array([float(w) for w in wave_cols])  # nm values

    X_cal = df_cal[wave_cols].values.astype(float)   # shape (273, 4200)
    X_val = df_val[wave_cols].values.astype(float)   # shape (120, 4200)

    y_cal = info_cal['Glucose (mM)'].values.astype(float)
    y_val = info_val['Glucose (mM)'].values.astype(float)

    # Drop rows with NaN glucose (none expected, but just in case)
    cal_ok = ~np.isnan(y_cal)
    val_ok = ~np.isnan(y_val)
    X_cal, y_cal, info_cal = X_cal[cal_ok], y_cal[cal_ok], info_cal[cal_ok]
    X_val, y_val, info_val = X_val[val_ok], y_val[val_ok], info_val[val_ok]

    print(f"  Calibration : {X_cal.shape[0]} samples, "
          f"{X_cal.shape[1]} wavelength points")
    print(f"  Validation  : {X_val.shape[0]} samples, "
          f"{X_val.shape[1]} wavelength points")
    print(f"  Wavelength  : {wavelengths[0]:.1f} – {wavelengths[-1]:.1f} nm")

    return dict(
        X_cal=X_cal, y_cal=y_cal, info_cal=info_cal,
        X_val=X_val, y_val=y_val, info_val=info_val,
        wavelengths=wavelengths,
    )


def discretize_glucose(y_continuous):
    """
    Convert continuous glucose (mM) into class labels.

    Bins   : 0–10 → 'Low'  |  10–30 → 'Normal'  |  30–50 → 'High'
    Returns: numpy array of str labels, same length as y_continuous.
    """
    labels = np.empty(len(y_continuous), dtype=object)
    labels[y_continuous <= 10]                          = 'Low'
    labels[(y_continuous > 10) & (y_continuous <= 30)] = 'Normal'
    labels[y_continuous > 30]                           = 'High'
    return labels


def top_correlated_wavelengths(X, y, wavelengths, n=20):
    """
    Find the n wavelengths whose absorbance is most correlated
    with glucose concentration (by Pearson |r|).

    Parameters
    ----------
    X           : (n_samples, 4200) spectra
    y           : (n_samples,) glucose values
    wavelengths : (4200,) nm array
    n           : how many top wavelengths to return

    Returns
    -------
    top_idx : indices into the 4200-column axis
    top_nm  : corresponding wavelength values in nm
    top_r   : correlation coefficients
    """
    correlations = np.array([
        np.corrcoef(X[:, i], y)[0, 1]
        for i in range(X.shape[1])
    ])
    top_idx = np.argsort(np.abs(correlations))[-n:][::-1]
    return top_idx, wavelengths[top_idx], correlations[top_idx]