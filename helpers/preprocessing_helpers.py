import pandas as pd
from typing import List, Dict, Optional
import os

def load_competition_data(
    data_dir="data/dm-lab-2-private-competition",
    data_identification_file="data_identification.csv",
    emotion_file="emotion.csv",
    source_data_file="final_posts.json",
    synthetic_data_file="llm_generate_data.csv"
):
    """
    Loads the three competition files and returns them.
    
    Parameters
    ----------
    data_dir : str
        Base directory where all files are stored.
    data_identification_file : str
        Filename for data identification CSV.
    emotion_file : str
        Filename for emotion CSV.
    source_data_file : str
        Filename for source data JSON.
    synthetic_data_file : str
        Filename for synthetic data CSV.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        data_identification, emotion, source_data, synthetic_data DataFrames.
    """

    data_identification_path = os.path.join(data_dir, data_identification_file)
    emotion_path = os.path.join(data_dir, emotion_file)
    source_data_path = os.path.join(data_dir, source_data_file)
    synthetic_data_path = os.path.join(data_dir, synthetic_data_file)

    data_identification = pd.read_csv(data_identification_path)
    emotion = pd.read_csv(emotion_path)
    source_data = pd.read_json(source_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)

    return data_identification, emotion, source_data, synthetic_data


def normalize_and_extract(
    json_series: pd.Series,
    fields: List[str],
    rename_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Normalizes a column of JSON objects and extracts specific fields.

    Parameters
    ----------
    json_series : pd.Series
        A pandas Series containing JSON dictionaries. Example: source_data['root']

    fields : list[str]
        List of fields to extract after normalization. Example: ["_source.post.post_id", "_source.post.text"]

    rename_map : dict, optional
        Mapping of original field names to new names. Example: {"_source.post.post_id": "id"}

    Returns
    -------
    pd.DataFrame
        A cleaned dataframe with selected and optionally renamed columns.
    """
    # Flatten nested JSON
    df = pd.json_normalize(json_series)

    # Ensure all requested fields exist
    missing_fields = [f for f in fields if f not in df.columns]
    if missing_fields:
        raise KeyError(f"Fields not found in JSON: {missing_fields}")

    # Extract desired fields
    df = df[fields]
    
    # Rename columns if mapping provided
    if rename_map:
        df = rename_columns(df, rename_map)

    return df

def rename_columns(
    df: pd.DataFrame, 
    rename_map: Dict[str, str] = None
    
) -> pd.DataFrame: 
    """
        df : original dataframe that needs column renaming
        Mapping of original dataframe. Example: source_data

        rename_map : dict, optional
        Mapping of original field names to new names. Example: {"_source.post.post_id": "id"}

        Returns
        -------
        pd.DataFrame
            A cleaned dataframe with renamed columns as requested.

    """
    df = df.rename(columns=rename_map)
    return df

__all__ = [
    "load_competition_data",
    "normalize_and_extract",
    "rename_columns"
]