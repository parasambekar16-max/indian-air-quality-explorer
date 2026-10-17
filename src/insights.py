"""
Insights Module
===============
Generates data-driven insights automatically from computed statistics.
All insights are dynamically computed from the current dataset and filters.
No hardcoded findings.

Designed for students and analysts to quickly summarize environmental air quality patterns,
seasonal variations, pollutant inter-correlations, and data quality metrics.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Safe import to handle both package and standalone execution
try:
    from src.data_loader import POLLUTANT_COLUMNS, NUMERIC_COLUMNS
except ImportError:
    try:
        from data_loader import POLLUTANT_COLUMNS, NUMERIC_COLUMNS
    except ImportError:
        POLLUTANT_COLUMNS = [
            'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
            'Benzene', 'Toluene', 'Xylene'
        ]
        NUMERIC_COLUMNS = POLLUTANT_COLUMNS + ['AQI']


def generate_overview_insights(df: pd.DataFrame) -> List[str]:
    """
    Generate high-level overview insights from the dataset.
    Each insight is computed dynamically from actual data.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned or filtered air quality dataframe.

    Returns:
    --------
    List[str]: List of bulleted insight strings.
    """
    insights = []
    
    if df.empty:
        return ['No data available for analysis.']
    
    # Number of cities and observations
    if 'City' in df.columns:
        n_cities = df['City'].nunique()
        insights.append(f"📊 The dataset covers **{n_cities}** monitoring locations with **{len(df):,}** total observations.")
    else:
        insights.append(f"📊 The dataset contains **{len(df):,}** total observations.")
    
    # Date range
    if 'Date' in df.columns and df['Date'].notna().any():
        dates = pd.to_datetime(df['Date'].dropna())
        if not dates.empty:
            date_min = dates.min()
            date_max = dates.max()
            span = (date_max - date_min).days
            insights.append(
                f"📅 Data spans from **{date_min.strftime('%B %d, %Y')}** to "
                f"**{date_max.strftime('%B %d, %Y')}** ({span:,} days)."
            )
    
    # Overall data completeness
    total_cells = df.shape[0] * df.shape[1]
    if total_cells > 0:
        total_missing = df.isna().sum().sum()
        completeness = (1 - total_missing / total_cells) * 100
        insights.append(f"📋 Overall data completeness: **{completeness:.1f}%**")
    
    return insights


def generate_pollution_insights(df: pd.DataFrame) -> List[str]:
    """
    Generate pollution-level insights comparing cities and pollutants.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing pollutant measurements and city names.

    Returns:
    --------
    List[str]: Formatted pollution comparison insights.
    """
    insights = []
    
    if df.empty or 'City' not in df.columns:
        return insights
    
    # Highest pollution city for each available pollutant
    for pollutant in POLLUTANT_COLUMNS:
        if pollutant in df.columns and df[pollutant].notna().any():
            city_means = df.groupby('City')[pollutant].mean().dropna()
            
            if not city_means.empty:
                highest_city = city_means.idxmax()
                highest_val = city_means.max()
                lowest_city = city_means.idxmin()
                lowest_val = city_means.min()
                
                insights.append(
                    f"🏭 **{highest_city}** has the highest average recorded {pollutant} "
                    f"({highest_val:.1f}) in the selected dataset, while **{lowest_city}** "
                    f"has the lowest ({lowest_val:.1f})."
                )
    
    # AQI insights
    if 'AQI' in df.columns and df['AQI'].notna().any():
        avg_aqi = df['AQI'].mean()
        max_aqi = df['AQI'].max()
        max_idx = df['AQI'].idxmax()
        
        city_name = 'Unknown'
        if 'City' in df.columns:
            loc_val = df.loc[max_idx, 'City']
            city_name = loc_val.iloc[0] if isinstance(loc_val, pd.Series) else str(loc_val)
        
        insights.append(
            f"🌡️ Average AQI across the dataset: **{avg_aqi:.1f}**. "
            f"The highest recorded AQI is **{max_aqi:.0f}** "
            f"in **{city_name}**."
        )
    
    return insights


def generate_seasonal_insights(df: pd.DataFrame) -> List[str]:
    """
    Generate seasonal pattern insights focusing on particulate matter and weather variations.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'Season' and/or 'Month' and pollutant columns.

    Returns:
    --------
    List[str]: List of seasonal trend insights.
    """
    insights = []
    
    if df.empty:
        return insights
    
    # Seasonal breakdown for primary particulates
    if 'Season' in df.columns:
        for pollutant in ['PM2.5', 'PM10']:
            if pollutant in df.columns and df[pollutant].notna().any():
                seasonal = df.groupby('Season')[pollutant].mean().dropna()
                
                if not seasonal.empty:
                    worst_season = seasonal.idxmax()
                    best_season = seasonal.idxmin()
                    ratio = seasonal.max() / seasonal.min() if seasonal.min() > 0 else 0
                    
                    insights.append(
                        f"🍂 **{worst_season}** shows the highest average {pollutant} "
                        f"({seasonal.max():.1f}), which is **{ratio:.1f}x** the "
                        f"**{best_season}** average ({seasonal.min():.1f})."
                    )
    
    # Month with highest pollution
    if 'Month' in df.columns:
        for pollutant in ['PM2.5']:
            if pollutant in df.columns and df[pollutant].notna().any():
                monthly = df.groupby('Month')[pollutant].mean().dropna()
                if not monthly.empty:
                    month_names = {
                        1: 'January', 2: 'February', 3: 'March', 4: 'April',
                        5: 'May', 6: 'June', 7: 'July', 8: 'August',
                        9: 'September', 10: 'October', 11: 'November', 12: 'December'
                    }
                    worst_month_key = monthly.idxmax()
                    worst_month = month_names.get(worst_month_key, str(worst_month_key))
                    insights.append(
                        f"📆 **{worst_month}** shows the highest mean {pollutant} "
                        f"({monthly.max():.1f}) across all selected cities."
                    )
    
    return insights


def generate_correlation_insights(df: pd.DataFrame) -> List[str]:
    """
    Generate correlation insights between pollutants.
    Note: These describe observed correlations, not causal relationships.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing multiple pollutant measurements.

    Returns:
    --------
    List[str]: Correlation summary sentences.
    """
    insights = []
    
    available = [col for col in POLLUTANT_COLUMNS if col in df.columns and df[col].notna().any()]
    
    if len(available) < 2:
        insights.append("⚠️ Not enough pollutant columns with data for correlation analysis.")
        return insights
    
    corr = df[available].corr()
    
    # Extract non-diagonal pairwise correlations
    strong_pairs = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            val = corr.iloc[i, j]
            if not np.isnan(val):
                strong_pairs.append((available[i], available[j], float(val)))
    
    if strong_pairs:
        # Sort by absolute correlation magnitude
        strong_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        top = strong_pairs[0]
        strength = 'strong' if abs(top[2]) > 0.7 else 'moderate' if abs(top[2]) > 0.4 else 'weak'
        direction = 'positive' if top[2] > 0 else 'negative'
        
        insights.append(
            f"🔗 Strongest observed correlation: **{top[0]}** and **{top[1]}** "
            f"show a **{strength} {direction}** correlation (r = {top[2]:.3f}). "
            f"Note: Correlation does not imply causation."
        )
        
        if len(strong_pairs) > 1:
            second = strong_pairs[1]
            strength_sec = 'strong' if abs(second[2]) > 0.7 else 'moderate' if abs(second[2]) > 0.4 else 'weak'
            direction_sec = 'positive' if second[2] > 0 else 'negative'
            insights.append(
                f"🔗 Second strongest: **{second[0]}** and **{second[1]}** "
                f"show a **{strength_sec} {direction_sec}** correlation (r = {second[2]:.3f})."
            )
    
    return insights


def generate_data_quality_insights(df: pd.DataFrame) -> List[str]:
    """
    Generate data completeness and missingness insights.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with air quality measurements.

    Returns:
    --------
    List[str]: Observations on data coverage and completeness.
    """
    insights = []
    
    if df.empty:
        return ['No data available.']
    
    # Most and least complete pollutant columns
    completeness = {}
    for col in POLLUTANT_COLUMNS:
        if col in df.columns:
            pct = (1 - df[col].isna().sum() / len(df)) * 100
            completeness[col] = pct
    
    if completeness:
        most_complete = max(completeness, key=completeness.get)
        least_complete = min(completeness, key=completeness.get)
        
        insights.append(
            f"✅ Most complete pollutant column: **{most_complete}** ({completeness[most_complete]:.1f}% data available)."
        )
        insights.append(
            f"⚠️ Least complete pollutant column: **{least_complete}** ({completeness[least_complete]:.1f}% data available). "
            f"This has the largest proportion of missing observations."
        )
    
    # City with most missing data across pollutant columns
    if 'City' in df.columns:
        pollutant_cols = [col for col in POLLUTANT_COLUMNS if col in df.columns]
        if pollutant_cols:
            city_missing = df.groupby('City')[pollutant_cols].apply(
                lambda x: x.isna().sum().sum() / (len(x) * len(pollutant_cols)) * 100
            ).round(1)
            
            if not city_missing.empty:
                worst_city = city_missing.idxmax()
                insights.append(
                    f"📍 **{worst_city}** has the highest proportion of missing "
                    f"pollutant measurements ({city_missing.max():.1f}%)."
                )
    
    return insights


def generate_all_insights(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Generate all categories of dynamic insights.
    Returns a dictionary organized by category.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned and filtered dataset.

    Returns:
    --------
    Dict[str, List[str]]: Dictionary with insight lists categorized by theme.
    """
    return {
        'Overview': generate_overview_insights(df),
        'Pollution Levels': generate_pollution_insights(df),
        'Seasonal Patterns': generate_seasonal_insights(df),
        'Correlations': generate_correlation_insights(df),
        'Data Quality': generate_data_quality_insights(df)
    }
