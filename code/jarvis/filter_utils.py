import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def apply_filters(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply reproducible filters to JARVIS-DFT dataset based on config thresholds.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing candidate materials.
    config : dict
        Configuration dictionary parsed from TOML.
    
    Returns
    -------
    pd.DataFrame
        Filtered candidate materials.
    """
    # --- Extract filter parameters ---
    bandgap_col   = config["filters"]["bandgap_column"]
    sem_min       = config["filters"]["semiconductor_min"]
    sem_max       = config["filters"]["semiconductor_max"]
    trans_min     = config["filters"]["transparent_min"]
    toxic_elements = config["filters"]["toxic_elements"]

    logger.info(f"Using bandgap column: {bandgap_col}")
    logger.info(f"Applying filters: semiconductor [{sem_min}, {sem_max}], "
                f"transparent > {trans_min}, exclude {toxic_elements}")

    # --- Apply filters ---
    df = df.copy()  # avoid SettingWithCopyWarning
    df["is_semiconductor"] = df[bandgap_col].between(sem_min, sem_max)
    df["is_transparent"]   = df[bandgap_col] > trans_min
    df["is_nontoxic"]      = ~df["formula"].str.contains("|".join(toxic_elements), na=False)

    # --- Combine filters ---
    mask = df["is_semiconductor"] & df["is_transparent"] & df["is_nontoxic"]
    df_candidates = df[mask].reset_index(drop=True)

    logger.info(f"Candidate materials found: {df_candidates.shape[0]}")
    return df_candidates

