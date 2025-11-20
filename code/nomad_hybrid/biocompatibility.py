import pandas as pd 
import re 
import numpy as np
import logging


# Based on just OSHA
toxic_elements = = [
    'As',  # Arsenic
    'Pb',  # Lead
    'Hg',  # Mercury
    'Cd',  # Cadmium
    'Be',  # Beryllium
    'Ni',  # Nickel  
    'Cr',  # Chromium
    'Ba',  # Barium
]

logger = logging.getLogger(__name__)

def apply_filters(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Calculates a 'biocompatibility_score' for the JARVIS-DFT dataset.
    
    This function does NOT remove any rows. It adds a score column
    so that downstream models can learn the relationship between 
    toxicity/stability and the target properties.
    
    Parameters
    ----------
    df : pd.DataFrame
        
    config : dict
       
    
    Returns
    -------
    pd.DataFrame
        Original dataset with new 'biocompatibility_score' column.
    """
    # 1. Extract Parameters 
    toxic_elems = config["filters"]["toxic_elements"]
    formula_col = config["columns"]["formula"] 
    ehull_col = config["columns"]["ehull"]    
    
    # 2.Copy df
    df_scored = df.copy()
    
    
    # 3. Determine Toxicity (Flagging)
    # Matches element symbol ONLY if followed by a number, uppercase letter, or end of string.
    # This prevents matching 'P' (Phosphorus) inside 'Pb' (Lead).
    toxic_pattern = r'(' + r'|'.join([re.escape(el) + r'(\d|(?=[A-Z])|$)' for el in toxic_elements]) + r')'
    
    df_scored["has_toxic_element"] = df_scored[formula_col].str.contains(toxic_pattern, regex=True, na=False)
    
    # 4. Determine Stability 
    # Stable if energy above hull is near 0. We allow < 0.1 
    if ehull_col in df_scored.columns:
        df_scored["is_stable"] = df_scored[ehull_col] <= 0.10

    # 5. Calculate Composite Score (0-100)
    def calculate_bio_score(row):
        score = 100
        
        # A. Penalty for Toxic Elements (-50)
        if row["has_toxic_element"]:
            score -= 50
            # Redemption: If it's very stable\ (+20)
            if row["is_stable"]:
                score += 20
        
        # B. Penalty for Instability (-30)
        if not row["is_stable"]:
            score -= 30
            
        return max(0, score)

    df_scored["biocompatibility_score"] = df_scored.apply(calculate_bio_score, axis=1)
    
    return df_scored
    
    
    
    
    

'''
def filter_biocompatibility(df):
    
    """
    Filters through a data frame and removes compounds containing toxic elements. 
    
    The filter_biocompatibility function reads a data frame, removes any rows containing toxic elements
    (OSHA) based on chemical formula and returns a cleaned Dataframe. 
    
    Input:
        df : The input DataFrame (NOMAD/JARVIS .csv)

    Returns:
        pd.DataFrame: New DataFrame containing only biocompatible materials.
    """
    
    print("Biocompatibility Filter starts")
    
    
    #1 Find Column with Chemical Formula 
    formula_column = None
    if 'formula' in df.columns:
        formula_column = 'formula'
        
    if not formula_column:
        print("  [Filter Error] Could not find a 'formula' column.")
        print(f"  Available column names are: {df.columns.tolist()}")
        print("  Returning original DataFrame.")
        return df
    
    #2 Apply Filter using Regex 
    # Looks for element symbols, starting with capital letter
    
    toxic_pattern = r'(' + r'|'.join([re.escape(element) + r'(\d|(?=[A-Z])|$)' for element in toxic_elements]) + r')'
    
    toxic = df[formula_column].str.contains(toxic_pattern, regex=True, na=True) # NA = True 
    
    #3 Apply  Filter
    # Tilde (~) to *invert* the toxic (toxic--> False and non toxic to True)
    clean_df = df[~toxic] # Only keep biocompatible ones in clean_df

    original_count = len(df)
    clean_count = len(clean_df)
    removed_count = original_count - clean_count

    print(f"  ...Filter complete.")
    print(f"  Original materials: {original_count}")
    print(f"  Removed {removed_count} toxic materials.")
    print(f"  Biocompatible materials remaining: {clean_count}")

    return clean_df
    '''