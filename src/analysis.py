"""
Analysis Module
===============
Performs statistical analysis, aggregation, and correlation analysis
on air quality data using NumPy and Pandas.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    from src.data_loader import POLLUTANT_COLUMNS, NUMERIC_COLUMNS
except ImportError:
    from data_loader import POLLUTANT_COLUMNS, NUMERIC_COLUMNS


def get_descriptive_stats(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate descriptive statistics for numeric columns.
    Uses NumPy for statistical calculations.
    """
    if df.empty:
        return pd.DataFrame()
    
    if columns is None:
        columns = [col for col in NUMERIC_COLUMNS if col in df.columns]
    
    stats_list = []
    for col in columns:
        if col in df.columns and df[col].notna().any():
            values = df[col].dropna().values  # NumPy array
            stats_list.append({
                'Pollutant': col,
                'Count': int(len(values)),
                'Mean': round(float(np.mean(values)), 2),
                'Median': round(float(np.median(values)), 2),
                'Std Dev': round(float(np.std(values, ddof=1)), 2),
                'Min': round(float(np.min(values)), 2),
                '25th Percentile': round(float(np.percentile(values, 25)), 2),
                '75th Percentile': round(float(np.percentile(values, 75)), 2),
                '90th Percentile': round(float(np.percentile(values, 90)), 2),
                '95th Percentile': round(float(np.percentile(values, 95)), 2),
                'Max': round(float(np.max(values)), 2),
                'Missing': int(df[col].isna().sum()),
                'Completeness %': round((1 - df[col].isna().sum() / len(df)) * 100, 1)
            })
    
    return pd.DataFrame(stats_list)


def get_city_stats(df: pd.DataFrame, pollutant: str = 'PM2.5') -> pd.DataFrame:
    """
    Get city-wise statistics for a specific pollutant.
    Uses Pandas groupby for aggregation.
    """
    if df.empty or 'City' not in df.columns or pollutant not in df.columns:
        return pd.DataFrame()
    
    # Pandas groupby with multiple aggregation functions
    city_stats = df.groupby('City')[pollutant].agg(
        ['count', 'mean', 'median', 'std', 'min', 'max']
    ).round(2)
    
    # Add percentiles using NumPy
    p90 = df.groupby('City')[pollutant].apply(
        lambda x: round(float(np.percentile(x.dropna(), 90)), 2) if x.notna().any() else np.nan
    )
    city_stats['p90'] = p90
    
    city_stats = city_stats.reset_index()
    city_stats.columns = ['City', 'Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', '90th Pctl']
    city_stats = city_stats.sort_values('Mean', ascending=False)
    
    return city_stats


def normalize_resample_freq(freq: str) -> str:
    """Normalize frequency aliases for pandas resample (e.g., M -> ME, Y -> YE in modern pandas)."""
    freq_map = {
        'M': 'ME', 'm': 'ME',
        'Y': 'YE', 'y': 'YE',
        'A': 'YE', 'a': 'YE',
        'Q': 'QE', 'q': 'QE',
        'W': 'W', 'w': 'W',
        'D': 'D', 'd': 'D'
    }
    return freq_map.get(freq, freq)


def aggregate_time_series(df: pd.DataFrame, freq: str = 'M', 
                          cities: Optional[List[str]] = None,
                          pollutant: str = 'PM2.5') -> pd.DataFrame:
    """
    Aggregate pollution data by time frequency.
    freq: 'D' (daily), 'W' (weekly), 'M' (monthly), 'Q' (quarterly), 'Y' (yearly)
    """
    if df.empty or 'Date' not in df.columns or pollutant not in df.columns:
        return pd.DataFrame()
    
    data = df.copy()
    
    if cities:
        data = data[data['City'].isin(cities)]
    
    if data.empty:
        return pd.DataFrame()
    
    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    
    resample_freq = normalize_resample_freq(freq)
    
    # Pandas resample for time-based grouping with version-compatible frequency fallback
    try:
        if 'City' in data.columns:
            grouped = data.set_index('Date').groupby('City')[pollutant].resample(resample_freq).mean().reset_index()
        else:
            grouped = data.set_index('Date')[pollutant].resample(resample_freq).mean().reset_index()
    except ValueError:
        if 'City' in data.columns:
            grouped = data.set_index('Date').groupby('City')[pollutant].resample(freq).mean().reset_index()
        else:
            grouped = data.set_index('Date')[pollutant].resample(freq).mean().reset_index()
    
    grouped[pollutant] = grouped[pollutant].round(2)
    
    return grouped


def get_monthly_stats(df: pd.DataFrame, pollutant: str = 'PM2.5',
                      cities: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Get month-wise statistics for seasonal analysis.
    """
    if df.empty or 'Month' not in df.columns or pollutant not in df.columns:
        return pd.DataFrame()
    
    data = df.copy()
    if cities:
        data = data[data['City'].isin(cities)]
    
    month_stats = data.groupby('Month')[pollutant].agg(['mean', 'median', 'std', 'count']).round(2)
    month_stats = month_stats.reset_index()
    
    # Add month names
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    month_stats['Month_Name'] = month_stats['Month'].map(month_names)
    
    return month_stats


def get_seasonal_stats(df: pd.DataFrame, pollutant: str = 'PM2.5',
                       cities: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Get season-wise statistics.
    Seasons: Winter (Dec-Feb), Summer (Mar-May), Monsoon (Jun-Sep), Post-Monsoon (Oct-Nov)
    """
    if df.empty or 'Season' not in df.columns or pollutant not in df.columns:
        return pd.DataFrame()
    
    data = df.copy()
    if cities:
        data = data[data['City'].isin(cities)]
    
    season_order = ['Winter', 'Summer', 'Monsoon', 'Post-Monsoon']
    seasonal = data.groupby('Season')[pollutant].agg(['mean', 'median', 'std', 'count']).round(2)
    seasonal = seasonal.reindex(season_order)
    seasonal = seasonal.reset_index()
    
    return seasonal


def get_city_seasonal_pivot(df: pd.DataFrame, pollutant: str = 'PM2.5') -> pd.DataFrame:
    """
    Create a city x month pivot table for heatmap visualization.
    """
    if df.empty or 'City' not in df.columns or 'Month' not in df.columns:
        return pd.DataFrame()
    
    if pollutant not in df.columns:
        return pd.DataFrame()
    
    # Pandas pivot_table: cities as rows, months as columns
    pivot = df.pivot_table(
        values=pollutant, 
        index='City', 
        columns='Month', 
        aggfunc='mean'
    ).round(1)
    
    # Rename columns to month names
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    pivot = pivot.rename(columns=month_names)
    
    return pivot


def get_correlation_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate correlation matrix for pollutant columns.
    Uses Pandas .corr() which relies on NumPy internally.
    """
    if df.empty:
        return pd.DataFrame()
    
    if columns is None:
        columns = [col for col in NUMERIC_COLUMNS if col in df.columns and df[col].notna().any()]
    
    if len(columns) < 2:
        return pd.DataFrame()
    
    correlation = df[columns].corr().round(3)
    return correlation


def get_top_bottom_cities(df: pd.DataFrame, pollutant: str = 'PM2.5', 
                          n: int = 5, metric: str = 'mean') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get top N most and least polluted cities by a given pollutant.
    Note: Rankings are only within the available dataset.
    """
    if df.empty or 'City' not in df.columns or pollutant not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    
    city_means = df.groupby('City')[pollutant].agg(metric).round(2).reset_index()
    city_means.columns = ['City', f'{metric.title()} {pollutant}']
    city_means = city_means.sort_values(f'{metric.title()} {pollutant}', ascending=False)
    
    top_n = city_means.head(n)
    bottom_n = city_means.tail(n).sort_values(f'{metric.title()} {pollutant}')
    
    return top_n, bottom_n


def get_yearly_trend(df: pd.DataFrame, pollutant: str = 'PM2.5',
                     cities: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Get year-wise trend for analysis.
    """
    if df.empty or 'Year' not in df.columns or pollutant not in df.columns:
        return pd.DataFrame()
    
    data = df.copy()
    if cities:
        data = data[data['City'].isin(cities)]
    
    yearly = data.groupby(['Year', 'City'])[pollutant].agg(['mean', 'count']).round(2).reset_index()
    yearly.columns = ['Year', 'City', f'Mean {pollutant}', 'Observations']
    
    return yearly


def get_aqi_distribution(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Get AQI bucket distribution if AQI data is available.
    """
    if 'AQI_Bucket' not in df.columns or df['AQI_Bucket'].isna().all():
        return None
    
    distribution = df['AQI_Bucket'].value_counts().reset_index()
    distribution.columns = ['AQI Category', 'Count']
    distribution['Percentage'] = (distribution['Count'] / distribution['Count'].sum() * 100).round(1)
    
    # Order by severity
    order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    distribution['sort_key'] = distribution['AQI Category'].apply(
        lambda x: order.index(x) if x in order else 99
    )
    distribution = distribution.sort_values('sort_key').drop('sort_key', axis=1).reset_index(drop=True)
    
    return distribution
