"""
Data Loader Module
==================
Handles loading air quality data from CSV files.
Supports the bundled demo dataset and user-uploaded CSVs.
"""
import pandas as pd
import numpy as np
import os
import streamlit as st
from typing import Optional, Tuple, Dict, List

# Define expected column schemas
EXPECTED_COLUMNS = {
    'city': ['city', 'City', 'CITY', 'city_name', 'location', 'Location', 'station', 'Station'],
    'date': ['date', 'Date', 'DATE', 'datetime', 'Datetime', 'sampling_date', 'Sampling Date'],
    'pm25': ['pm2.5', 'PM2.5', 'pm25', 'PM25', 'pm2_5', 'PM2_5', 'Fine Particulate Matter'],
    'pm10': ['pm10', 'PM10', 'Pm10', 'Particulate Matter'],
    'no2': ['no2', 'NO2', 'No2', 'Nitrogen Dioxide'],
    'so2': ['so2', 'SO2', 'So2', 'Sulphur Dioxide', 'Sulfur Dioxide'],
    'co': ['co', 'CO', 'Co', 'Carbon Monoxide'],
    'o3': ['o3', 'O3', 'Ozone'],
    'aqi': ['aqi', 'AQI', 'Aqi', 'Air Quality Index'],
    'aqi_bucket': ['aqi_bucket', 'AQI_Bucket', 'AQI_Category', 'aqi_category', 'Bucket']
}

POLLUTANT_COLUMNS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
NUMERIC_COLUMNS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']

def get_default_data_path() -> str:
    """Return path to the bundled demonstration dataset."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'data', 'sample_air_quality.csv')

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map various column name aliases to standard names."""
    # Create a copy
    df = df.copy()
    column_mapping = {}
    
    for standard_name, aliases in EXPECTED_COLUMNS.items():
        for col in df.columns:
            if col.strip() in aliases:
                # Map to our standard names
                standard_mapped = {
                    'city': 'City', 'date': 'Date', 'pm25': 'PM2.5', 'pm10': 'PM10',
                    'no2': 'NO2', 'so2': 'SO2', 'co': 'CO', 'o3': 'O3',
                    'aqi': 'AQI', 'aqi_bucket': 'AQI_Bucket'
                }
                if col.strip() != standard_mapped[standard_name]:
                    column_mapping[col.strip()] = standard_mapped[standard_name]
                break
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    return df

def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    """Validate the dataset has minimum required columns.
    Returns (is_valid, found_columns, missing_required).
    """
    required = ['City', 'Date']
    found = [col for col in required if col in df.columns]
    missing = [col for col in required if col not in df.columns]
    
    # Check for at least one pollutant column
    pollutant_found = [col for col in POLLUTANT_COLUMNS if col in df.columns]
    has_pollutants = len(pollutant_found) > 0
    
    is_valid = len(missing) == 0 and has_pollutants
    
    warnings = []
    if not has_pollutants:
        missing.append('At least one pollutant (PM2.5, PM10, NO2, SO2, CO, O3)')
    if 'AQI' not in df.columns:
        warnings.append('AQI column not found. AQI-specific features will be disabled.')
    
    return is_valid, found + pollutant_found, missing

@st.cache_data(ttl=3600)
def load_data(file_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """Load air quality data from CSV.
    Returns the DataFrame and a metadata dict with loading info.
    """
    metadata = {
        'source': 'unknown',
        'original_rows': 0,
        'original_columns': [],
        'column_mapping': {},
        'warnings': [],
        'errors': []
    }
    
    if file_path is None:
        file_path = get_default_data_path()
        metadata['source'] = 'bundled_demo'
    else:
        metadata['source'] = 'user_upload'
    
    try:
        # Try reading with common encodings
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin-1')
        
        metadata['original_rows'] = len(df)
        metadata['original_columns'] = list(df.columns)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Normalize column names
        df = normalize_columns(df)
        
        # Validate
        is_valid, found, missing = validate_dataset(df)
        
        if not is_valid:
            metadata['errors'].append(
                f"Dataset is missing required columns: {', '.join(missing)}. "
                f"Required: City/Location column, Date column, and at least one pollutant column."
            )
            return pd.DataFrame(), metadata
        
        if 'AQI' not in df.columns:
            metadata['warnings'].append('AQI column not found. AQI-specific features will be disabled.')
        
        return df, metadata
        
    except FileNotFoundError:
        metadata['errors'].append(f"File not found: {file_path}")
        return pd.DataFrame(), metadata
    except pd.errors.EmptyDataError:
        metadata['errors'].append("The uploaded file is empty.")
        return pd.DataFrame(), metadata
    except Exception as e:
        metadata['errors'].append(f"Error loading data: {str(e)}")
        return pd.DataFrame(), metadata


def load_uploaded_file(uploaded_file) -> Tuple[pd.DataFrame, Dict]:
    """Load data from a Streamlit uploaded file object."""
    import tempfile
    metadata = {
        'source': 'user_upload',
        'original_rows': 0,
        'original_columns': [],
        'warnings': [],
        'errors': []
    }
    
    try:
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            metadata['errors'].append("Unsupported file format. Please upload a CSV or Excel file.")
            return pd.DataFrame(), metadata
        
        metadata['original_rows'] = len(df)
        metadata['original_columns'] = list(df.columns)
        
        df.columns = df.columns.str.strip()
        df = normalize_columns(df)
        
        is_valid, found, missing = validate_dataset(df)
        if not is_valid:
            metadata['errors'].append(
                f"Dataset is missing required columns: {', '.join(missing)}. "
                f"Required: City/Location, Date, and at least one pollutant."
            )
            return pd.DataFrame(), metadata
        
        if 'AQI' not in df.columns:
            metadata['warnings'].append('AQI column not found. AQI-specific features will be disabled.')
        
        return df, metadata
    except Exception as e:
        metadata['errors'].append(f"Error reading uploaded file: {str(e)}")
        return pd.DataFrame(), metadata
