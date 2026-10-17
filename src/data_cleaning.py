"""
Data Cleaning Module
====================
Handles cleaning, validation, and preprocessing of air quality data.
Reports all cleaning decisions transparently.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from src.data_loader import POLLUTANT_COLUMNS, NUMERIC_COLUMNS
except ImportError:
    from data_loader import POLLUTANT_COLUMNS, NUMERIC_COLUMNS


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean the air quality dataset.
    Returns cleaned DataFrame and a cleaning report dictionary.
    
    Cleaning steps:
    1. Convert date column to datetime
    2. Convert numeric columns to proper types
    3. Detect and report duplicates
    4. Flag suspicious values (but don't remove valid extremes)
    5. Sort by city and date
    """
    if df.empty:
        return df, {'steps': [], 'rows_before': 0, 'rows_after': 0}
    
    report = {
        'steps': [],
        'rows_before': len(df),
        'rows_after': 0,
        'duplicates_found': 0,
        'duplicates_removed': 0,
        'invalid_dates': 0,
        'non_numeric_converted': {},
        'suspicious_values': {},
        'missing_before': {},
        'missing_after': {}
    }
    
    cleaned = df.copy()
    
    # Step 1: Record missing values before cleaning
    for col in cleaned.columns:
        missing_count = cleaned[col].isna().sum()
        if missing_count > 0:
            report['missing_before'][col] = int(missing_count)
    
    # Step 2: Date conversion
    if 'Date' in cleaned.columns:
        # Try multiple date formats
        original_non_null = cleaned['Date'].notna().sum()
        cleaned['Date'] = pd.to_datetime(cleaned['Date'], errors='coerce', dayfirst=False)
        
        # Try dayfirst=True for dates that failed
        null_after = cleaned['Date'].isna().sum()
        invalid_dates = null_after - df['Date'].isna().sum()
        
        if invalid_dates > 0:
            report['invalid_dates'] = int(invalid_dates)
            report['steps'].append(
                f"Date conversion: {invalid_dates} values could not be parsed as dates."
            )
        else:
            report['steps'].append("All date values parsed successfully.")
    
    # Step 3: Numeric conversion for pollutant and AQI columns
    for col in NUMERIC_COLUMNS:
        if col in cleaned.columns:
            non_numeric_before = 0
            if not pd.api.types.is_numeric_dtype(cleaned[col]):
                # Count non-numeric entries before coercion
                temp = pd.to_numeric(cleaned[col], errors='coerce')
                non_numeric_before = int(temp.isna().sum() - cleaned[col].isna().sum())
                cleaned[col] = temp
                
                if non_numeric_before > 0:
                    report['non_numeric_converted'][col] = non_numeric_before
                    report['steps'].append(
                        f"{col}: {non_numeric_before} non-numeric values converted to NaN."
                    )
            else:
                cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
    
    # Step 4: Detect duplicates
    duplicates = cleaned.duplicated()
    dup_count = int(duplicates.sum())
    report['duplicates_found'] = dup_count
    
    if dup_count > 0:
        cleaned = cleaned.drop_duplicates()
        report['duplicates_removed'] = dup_count
        report['steps'].append(
            f"Found and removed {dup_count} exact duplicate rows."
        )
    
    # Step 5: Flag suspicious values (but don't remove them!)
    # Very high pollution can be real in India (e.g., Delhi in winter)
    suspicious_thresholds = {
        'PM2.5': {'min': 0, 'max': 1000, 'note': 'Values above 500 µg/m³ are extreme but can occur'},
        'PM10': {'min': 0, 'max': 2000, 'note': 'Values above 1000 µg/m³ are extreme but can occur'},
        'NO2': {'min': 0, 'max': 500},
        'SO2': {'min': 0, 'max': 500},
        'CO': {'min': 0, 'max': 50},
        'O3': {'min': 0, 'max': 500},
        'AQI': {'min': 0, 'max': 500, 'note': 'AQI scale in India goes up to 500+'}
    }
    
    for col, thresholds in suspicious_thresholds.items():
        if col in cleaned.columns:
            negative = (cleaned[col] < thresholds['min']).sum()
            very_high = (cleaned[col] > thresholds['max']).sum()
            
            if negative > 0:
                # Negative values are clearly invalid for concentrations
                cleaned.loc[cleaned[col] < thresholds['min'], col] = np.nan
                report['suspicious_values'][col] = {
                    'negative_removed': int(negative)
                }
                report['steps'].append(
                    f"{col}: {negative} negative values set to NaN (invalid for concentrations)."
                )
            
            if very_high > 0:
                # Don't remove high values, just flag them
                note = thresholds.get('note', 'Extreme values flagged but retained')
                if col not in report['suspicious_values']:
                    report['suspicious_values'][col] = {}
                report['suspicious_values'][col]['extreme_high_flagged'] = int(very_high)
                report['steps'].append(
                    f"{col}: {very_high} values above {thresholds['max']} flagged (retained). {note}"
                )
    
    # Step 6: Strip whitespace from string columns
    if 'City' in cleaned.columns:
        cleaned['City'] = cleaned['City'].astype(str).str.strip()
    
    if 'AQI_Bucket' in cleaned.columns:
        cleaned['AQI_Bucket'] = cleaned['AQI_Bucket'].astype(str).str.strip()
        # Replace 'nan' string with actual NaN
        cleaned.loc[cleaned['AQI_Bucket'] == 'nan', 'AQI_Bucket'] = np.nan
    
    # Step 7: Sort by City and Date
    if 'City' in cleaned.columns and 'Date' in cleaned.columns:
        cleaned = cleaned.sort_values(['City', 'Date']).reset_index(drop=True)
        report['steps'].append("Data sorted by City and Date.")
    
    # Step 8: Add derived columns
    if 'Date' in cleaned.columns:
        cleaned['Year'] = cleaned['Date'].dt.year
        cleaned['Month'] = cleaned['Date'].dt.month
        cleaned['Month_Name'] = cleaned['Date'].dt.strftime('%B')
        cleaned['Day_of_Week'] = cleaned['Date'].dt.day_name()
        
        # Season mapping for India
        # Winter: Dec-Feb, Summer: Mar-May, Monsoon: Jun-Sep, Post-Monsoon: Oct-Nov
        season_map = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Summer', 4: 'Summer', 5: 'Summer',
            6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
            10: 'Post-Monsoon', 11: 'Post-Monsoon'
        }
        cleaned['Season'] = cleaned['Month'].map(season_map)
        report['steps'].append(
            "Added derived columns: Year, Month, Month_Name, Day_of_Week, Season. "
            "Seasons defined as: Winter (Dec-Feb), Summer (Mar-May), "
            "Monsoon (Jun-Sep), Post-Monsoon (Oct-Nov)."
        )
    
    # Record missing values after cleaning
    for col in cleaned.columns:
        missing_count = cleaned[col].isna().sum()
        if missing_count > 0:
            report['missing_after'][col] = int(missing_count)
    
    report['rows_after'] = len(cleaned)
    
    if len(report['steps']) == 0:
        report['steps'].append("No cleaning operations were needed.")
    
    return cleaned, report


def get_data_quality_report(df: pd.DataFrame) -> Dict:
    """
    Generate a comprehensive data quality report.
    """
    if df.empty:
        return {'error': 'Empty dataset'}
    
    report = {}
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # Overall stats
    report['total_rows'] = total_rows
    report['total_columns'] = total_cols
    report['total_cells'] = total_rows * total_cols
    
    # Missing values
    missing_info = {}
    for col in df.columns:
        missing = int(df[col].isna().sum())
        pct = round(missing / total_rows * 100, 2)
        missing_info[col] = {'count': missing, 'percentage': pct}
    report['missing_values'] = missing_info
    
    # Overall completeness
    total_missing = df.isna().sum().sum()
    report['overall_completeness'] = round((1 - total_missing / (total_rows * total_cols)) * 100, 2)
    
    # Duplicates
    report['duplicate_rows'] = int(df.duplicated().sum())
    
    # Date range
    if 'Date' in df.columns and df['Date'].notna().any():
        report['date_range'] = {
            'start': str(df['Date'].min().date()),
            'end': str(df['Date'].max().date()),
            'span_days': int((df['Date'].max() - df['Date'].min()).days)
        }
    
    # City coverage
    if 'City' in df.columns:
        city_counts = df['City'].value_counts()
        report['cities'] = {
            'count': int(df['City'].nunique()),
            'names': list(city_counts.index),
            'observation_counts': city_counts.to_dict()
        }
    
    # Numeric column stats
    numeric_stats = {}
    for col in NUMERIC_COLUMNS:
        if col in df.columns and df[col].notna().any():
            vals = df[col].dropna()
            numeric_stats[col] = {
                'count': int(vals.count()),
                'mean': round(float(vals.mean()), 2),
                'std': round(float(vals.std()), 2),
                'min': round(float(vals.min()), 2),
                'p25': round(float(np.percentile(vals, 25)), 2),
                'median': round(float(vals.median()), 2),
                'p75': round(float(np.percentile(vals, 75)), 2),
                'p90': round(float(np.percentile(vals, 90)), 2),
                'p95': round(float(np.percentile(vals, 95)), 2),
                'max': round(float(vals.max()), 2),
                'zeros': int((vals == 0).sum()),
                'missing': int(df[col].isna().sum()),
                'missing_pct': round(df[col].isna().sum() / total_rows * 100, 2)
            }
    report['numeric_stats'] = numeric_stats
    
    # Coverage by city and time
    if 'City' in df.columns and 'Date' in df.columns:
        coverage = {}
        for city in df['City'].unique():
            city_data = df[df['City'] == city]
            if city_data['Date'].notna().any():
                coverage[city] = {
                    'observations': len(city_data),
                    'date_start': str(city_data['Date'].min().date()) if city_data['Date'].notna().any() else 'N/A',
                    'date_end': str(city_data['Date'].max().date()) if city_data['Date'].notna().any() else 'N/A',
                    'pollutants_available': [
                        col for col in POLLUTANT_COLUMNS 
                        if col in city_data.columns and city_data[col].notna().any()
                    ]
                }
        report['city_coverage'] = coverage
    
    return report


def get_zero_vs_missing_distinction(df: pd.DataFrame) -> Dict:
    """
    Distinguish between zero readings and missing values.
    Zero can be a valid measurement, missing means no data.
    """
    result = {}
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            zeros = int((df[col] == 0).sum())
            missing = int(df[col].isna().sum())
            result[col] = {
                'zero_readings': zeros,
                'missing_values': missing,
                'note': 'Zero may be valid (very low concentration) or may indicate sensor issues. Missing means no measurement was recorded.'
            }
    return result
