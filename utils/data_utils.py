"""
Utility functions for data handling
"""

import json
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Union
from pathlib import Path

logger = logging.getLogger("utils.data")

def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load data from a JSON file
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data
    """
    file_path = Path(file_path)
    logger.debug(f"Loading JSON from {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {str(e)}")
        raise
        
def save_json(data: Any, file_path: Union[str, Path]) -> None:
    """
    Save data to a JSON file
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
    """
    file_path = Path(file_path)
    logger.debug(f"Saving JSON to {file_path}")
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {str(e)}")
        raise
        
def save_text(text: str, file_path: Union[str, Path]) -> None:
    """
    Save text to a file
    
    Args:
        text: Text to save
        file_path: Path to save the text file
    """
    file_path = Path(file_path)
    logger.debug(f"Saving text to {file_path}")
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        logger.error(f"Error saving text to {file_path}: {str(e)}")
        raise
        
def load_dataframe(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from a CSV or Excel file into a pandas DataFrame
    
    Args:
        file_path: Path to the file
        
    Returns:
        Loaded DataFrame
    """
    file_path = Path(file_path)
    logger.debug(f"Loading DataFrame from {file_path}")
    
    try:
        extension = file_path.suffix.lower()
        
        if extension == '.csv':
            df = pd.read_csv(file_path)
        elif extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
            
        return df
    except Exception as e:
        logger.error(f"Error loading DataFrame from {file_path}: {str(e)}")
        raise
        
def save_dataframe(df: pd.DataFrame, file_path: Union[str, Path], index: bool = False) -> None:
    """
    Save a pandas DataFrame to a file
    
    Args:
        df: DataFrame to save
        file_path: Path to save the file
        index: Whether to include the index in the output
    """
    file_path = Path(file_path)
    logger.debug(f"Saving DataFrame to {file_path}")
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        extension = file_path.suffix.lower()
        
        if extension == '.csv':
            df.to_csv(file_path, index=index)
        elif extension in ['.xlsx', '.xls']:
            df.to_excel(file_path, index=index)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to {file_path}: {str(e)}")
        raise
        
def merge_datasets(dataframes: List[pd.DataFrame], on: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    Merge multiple datasets into a single DataFrame
    
    Args:
        dataframes: List of DataFrames to merge
        on: Column(s) to merge on
        
    Returns:
        Merged DataFrame
    """
    logger.debug(f"Merging {len(dataframes)} datasets")
    
    if not dataframes:
        return pd.DataFrame()
        
    if len(dataframes) == 1:
        return dataframes[0]
        
    try:
        result = dataframes[0]
        
        for i, df in enumerate(dataframes[1:], 1):
            if on:
                result = pd.merge(result, df, on=on, how='outer')
            else:
                # Try to find common columns
                common_cols = list(set(result.columns) & set(df.columns))
                
                if common_cols:
                    result = pd.merge(result, df, on=common_cols, how='outer')
                else:
                    # No common columns, just concatenate
                    logger.warning(f"No common columns found for DataFrame {i}, concatenating instead")
                    result = pd.concat([result, df], axis=1)
                    
        return result
    except Exception as e:
        logger.error(f"Error merging datasets: {str(e)}")
        raise
        
def process_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Process missing values in a DataFrame
    
    Args:
        df: DataFrame to process
        strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop', 'zero')
        
    Returns:
        Processed DataFrame
    """
    logger.debug(f"Processing missing values using strategy: {strategy}")
    
    try:
        result = df.copy()
        
        if strategy == 'drop':
            # Drop rows with any missing values
            result = result.dropna()
        else:
            # Process numeric and non-numeric columns separately
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if result[col].isnull().any():
                    if strategy == 'mean':
                        result[col] = result[col].fillna(result[col].mean())
                    elif strategy == 'median':
                        result[col] = result[col].fillna(result[col].median())
                    elif strategy == 'zero':
                        result[col] = result[col].fillna(0)
                    elif strategy == 'mode':
                        result[col] = result[col].fillna(result[col].mode()[0])
            
            # For non-numeric columns, use mode or empty string
            non_numeric_cols = result.select_dtypes(exclude=[np.number]).columns
            
            for col in non_numeric_cols:
                if result[col].isnull().any():
                    if strategy == 'mode':
                        result[col] = result[col].fillna(result[col].mode()[0] if not result[col].mode().empty else "")
                    else:
                        result[col] = result[col].fillna("")
                        
        return result
    except Exception as e:
        logger.error(f"Error processing missing values: {str(e)}")
        raise
        
def detect_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, List[int]]:
    """
    Detect outliers in numeric columns of a DataFrame
    
    Args:
        df: DataFrame to analyze
        method: Method for detecting outliers ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Dictionary mapping column names to lists of outlier indices
    """
    logger.debug(f"Detecting outliers using method: {method}")
    
    try:
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            
            elif method == 'zscore':
                # Z-score method
                mean = df[col].mean()
                std = df[col].std()
                
                z_scores = abs((df[col] - mean) / std)
                outlier_indices = df[z_scores > threshold].index.tolist()
            
            else:
                raise ValueError(f"Unsupported outlier detection method: {method}")
                
            if outlier_indices:
                outliers[col] = outlier_indices
                
        return outliers
    except Exception as e:
        logger.error(f"Error detecting outliers: {str(e)}")
        raise
        
def summarize_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of a DataFrame
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary with dataset summary
    """
    logger.debug("Generating dataset summary")
    
    try:
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": {col: int(df[col].isnull().sum()) for col in df.columns},
            "numeric_summary": {},
            "categorical_summary": {}
        }
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary["numeric_summary"][col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std())
            }
            
        # Categorical columns summary
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts().head(10).to_dict()
            total_unique = df[col].nunique()
            
            summary["categorical_summary"][col] = {
                "unique_values": int(total_unique),
                "top_values": {str(k): int(v) for k, v in value_counts.items()}
            }
            
        return summary
    except Exception as e:
        logger.error(f"Error summarizing dataset: {str(e)}")
        raise