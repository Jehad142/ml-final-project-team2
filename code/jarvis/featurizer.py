import toml
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Utility functions ---

def normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sentinel strings with NaN and infer dtypes."""
    df = df.replace({"na": np.nan, "NA": np.nan, "NaN": np.nan, "": np.nan})
    df = df.infer_objects(copy=False)
    return df

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def drop_empty_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop numeric columns that are entirely NaN; keep categorical/object columns."""
    keep = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            if s.notna().any():
                keep.append(col)
        else:
            if s.dtype == "object":
                keep.append(col)
    return df[keep]

def flatten_dict_column(df, col, prefix):
    """Expand a dict-like column into separate numeric columns."""
    expanded = df[col].apply(lambda d: d if isinstance(d, dict) else {})
    expanded_df = pd.json_normalize(expanded)
    for c in expanded_df.columns:
        expanded_df[c] = pd.to_numeric(expanded_df[c], errors="coerce")
    expanded_df = expanded_df.add_prefix(f"{prefix}_")
    return pd.concat([df.drop(columns=[col]), expanded_df], axis=1)

def safe_list_stats(val):
    """Compute mean/std/min/max for a list-like or dict-like, ignoring non-numeric."""
    nums = []
    if isinstance(val, dict):
        vals = val.values()
    elif isinstance(val, (list, tuple)):
        vals = val
    else:
        vals = [val]
    for v in vals:
        try:
            x = pd.to_numeric(v, errors="coerce")
            if pd.notna(x):
                nums.append(float(x))
        except Exception:
            continue
    if len(nums) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    return (np.mean(nums), np.std(nums), np.min(nums), np.max(nums))

def flatten_list_column(df, col, prefix):
    """Flatten a list-like column into numeric summary stats using safe_list_stats."""
    stats = df[col].apply(safe_list_stats)
    df[f"{prefix}_mean"] = stats.apply(lambda t: t[0])
    df[f"{prefix}_std"]  = stats.apply(lambda t: t[1])
    df[f"{prefix}_min"]  = stats.apply(lambda t: t[2])
    df[f"{prefix}_max"]  = stats.apply(lambda t: t[3])
    return df.drop(columns=[col])

def sanitize_features(df: pd.DataFrame, drop_cols=None) -> pd.DataFrame:
    """Two-pass sanitizer: flatten dicts, then lists, drop structural artifacts."""
    df = normalize_missing(df)
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    for col in list(df.columns):
        if df[col].apply(lambda v: isinstance(v, dict)).any():
            df = flatten_dict_column(df, col, col)
    for col in list(df.columns):
        if df[col].apply(lambda v: isinstance(v, (list, tuple))).any():
            df = flatten_list_column(df, col, col)
    return df

def drop_residual_non_scalars(df: pd.DataFrame) -> pd.DataFrame:
    """Final sweep: drop any columns that still contain lists or dicts."""
    to_drop = []
    for col in df.columns:
        if df[col].apply(lambda v: isinstance(v, (list, tuple, dict))).any():
            to_drop.append(col)
    return df.drop(columns=to_drop, errors="ignore")

# --- Featurizer class ---

class Featurizer:
    def __init__(self, plan_path: str):
        """
        Initialize featurizer with a TOML plan file.
        """
        self.plan = toml.load(plan_path)["features"]

    def apply_plan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply featurization plan to dataframe.
        """
        df = df.copy()
        df = normalize_missing(df)

        for feature, rule in self.plan.items():
            if feature not in df.columns:
                continue

            plan_type = rule.get("plan", "keep")
            notes = rule.get("notes", "")

            logger.info(f"Processing {feature} with plan={plan_type} ({notes})")

            if plan_type == "drop":
                df.drop(columns=[feature], inplace=True, errors="ignore")

            elif plan_type == "numeric":
                df[feature] = pd.to_numeric(df[feature], errors="coerce")

            elif plan_type == "categorical":
                df[feature] = df[feature].astype(str)

            elif plan_type == "flatten":
                if df[feature].apply(lambda v: isinstance(v, dict)).any():
                    df = flatten_dict_column(df, feature, feature)
                elif df[feature].apply(lambda v: isinstance(v, (list, tuple))).any():
                    df = flatten_list_column(df, feature, feature)
                else:
                    df[feature] = pd.to_numeric(df[feature], errors="coerce")

            elif plan_type == "combine":
                eps_cols = [c for c in ["epsx","epsy","epsz"] if c in df.columns]
                if eps_cols:
                    df[eps_cols] = df[eps_cols].apply(pd.to_numeric, errors="coerce")
                    df["eps_mean"] = df[eps_cols].apply(
                        lambda row: row.dropna().mean() if not row.dropna().empty else np.nan, axis=1
                    )
                    df["eps_std"] = df[eps_cols].apply(
                        lambda row: row.dropna().std() if not row.dropna().empty else np.nan, axis=1
                    )

            elif plan_type == "network":
                df.drop(columns=[feature], inplace=True, errors="ignore")
                logger.info(f"Feature {feature} reserved for graph-based featurization.")

        # Global sanitization passes
        df = sanitize_features(df)
        df = drop_residual_non_scalars(df)
        df = drop_empty_features(df)

        return df

