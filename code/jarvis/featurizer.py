import toml
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Featurizer:
    def __init__(self, plan_path: str):
        """
        Initialize featurizer with a TOML plan file.
        
        Parameters
        ----------
        plan_path : str
            Path to TOML file containing feature plan.
        """
        self.plan = toml.load(plan_path)["features"]

    def apply_plan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply featurization plan to dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        
        Returns
        -------
        pd.DataFrame
            Transformed dataframe.
        """
        df = df.copy()
        df.replace("na", np.nan, inplace=True)
        df.replace("", np.nan, inplace=True)
        df = df.infer_objects(copy=False)
        
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
                df[feature] = df[feature].apply(self._flatten_value)

            elif plan_type == "combine":
                eps_cols = [c for c in ["epsx","epsy","epsz"] if c in df.columns]
                if eps_cols:
                    df[eps_cols] = df[eps_cols].apply(pd.to_numeric, errors="coerce")
                    df["eps_mean"] = df[eps_cols].mean(axis=1, skipna=True, numeric_only=True)
                    df["eps_std"]  = df[eps_cols].std(axis=1, skipna=True, numeric_only=True)


            elif plan_type == "network":
                # Placeholder: structural features handled by GNN featurizer
                df.drop(columns=[feature], inplace=True, errors="ignore")
                logger.info(f"Feature {feature} reserved for graph-based featurization.")

        return df

    def _flatten_value(self, val):
        """Flatten dict/list values into scalars if possible."""
        if isinstance(val, dict):
            return np.mean([pd.to_numeric(v, errors="coerce") for v in val.values() if v != "na"])
        elif isinstance(val, (list, tuple)):
            return np.mean([pd.to_numeric(v, errors="coerce") for v in val if v != "na"])
        else:
            return pd.to_numeric(val, errors="coerce")


