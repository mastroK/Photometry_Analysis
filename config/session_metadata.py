"""
Per-mouse cohort metadata (DOB, DREADD treatment/viral expression), loaded
from a hand-maintained spreadsheet (see cohort_metadata_template.csv) and
looked up by Mouse ID. Age-at-recording is computed from DOB and the
session's own date (io_utils.raw_loader.parse_session_id) -- this metadata
sheet is the only external data source this pipeline needs beyond a
session's own trial_table.
"""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["Mouse ID", "Date of Birth", "Treatment", "Viral_Expression"]


def load_cohort_metadata(path):
    """Load the cohort metadata sheet (.csv or .xlsx, dispatched on suffix),
    parse Date of Birth, and index by Mouse ID.

    Treatment/Viral_Expression are read with pandas' default missing-value
    coercion disabled (keep_default_na=False, na_values only for a blank
    cell): "None"/"none" is a real Treatment category here (no drug given),
    not missing data, and pandas would otherwise silently collapse it to NaN.
    """
    path = Path(path)
    read_kwargs = dict(keep_default_na=False, na_values=[""])
    df = pd.read_excel(path, **read_kwargs) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path, **read_kwargs)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")

    df["Date of Birth"] = pd.to_datetime(df["Date of Birth"])
    return df.set_index("Mouse ID")


def get_mouse_metadata(metadata_df, mouse_id):
    """Return the metadata row for mouse_id as a dict."""
    if mouse_id not in metadata_df.index:
        raise KeyError(f"Mouse '{mouse_id}' not found in cohort metadata")
    return metadata_df.loc[mouse_id].to_dict()


def compute_age_days(dob, session_date_str):
    """session_date_str is the raw 'MMDDYY' directory-name string returned by
    io_utils.raw_loader.parse_session_id (e.g. "060223" -> 2023-06-02).
    Returns the animal's age in days at the time of recording.
    """
    session_date = pd.to_datetime(session_date_str, format="%m%d%y")
    return (session_date - pd.Timestamp(dob)).days
