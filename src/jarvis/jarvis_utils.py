# jarvis_utils.py
import os
import pandas as pd

def load_or_fetch_dataset(dataset_name: str, data_func, store_dir: str) -> pd.DataFrame:
    """
    Utility function to load a JARVIS dataset.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'dft_3d').
    data_func : callable
        Function to fetch dataset (e.g., data from JARVIS API).
    store_dir : str
        Directory where pickle files are stored.
    
    Returns
    -------
    pd.DataFrame
        Loaded dataset as a DataFrame.
    """
    # Construct pickle file path
    pkl_file = os.path.join(store_dir, f'jarvis_{dataset_name}.pkl')
    
    '''
    # Check if pickle exists
    if os.path.exists(pkl_file):
        print(f"Loading existing pickle file: {pkl_file}")
        df = pd.read_pickle(pkl_file)
    else:
        print(f"Pickle not found. Fetching dataset: {dataset_name}")
        dft_data = data_func(dataset_name, store_dir=store_dir)
        df = pd.DataFrame(dft_data)
        df.to_pickle(pkl_file)
        print(f"Dataset saved to {pkl_file}")
    '''

    # Always try to load pickle first, refetch on failure
    df = []
    try:
        df = pd.read_pickle(pkl_file)
    except Exception as e:
        print(f"Pickle load failed: {e}, refetching dataset...")
        dft_data = data_func(dataset_name, store_dir=store_dir)
        df = pd.DataFrame(dft_data)
        df.to_pickle(pkl_file)
        print(f"Dataset saved to {pkl_file}")
    
    print("Dataset shape:", df.shape)
    return df
