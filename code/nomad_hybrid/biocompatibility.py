import pandas as pd 
import re 

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
] # NEed to add to this list ?...  

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