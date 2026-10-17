"""
Visualization Module
====================
Creates interactive and static visualizations for air quality data.
Uses Plotly for interactive charts and Matplotlib/Seaborn for statistical plots.

This module is designed for students and researchers exploring Indian Air Quality datasets.
It provides consistent styling, color palettes according to Indian National Air Quality Index (NAQI)
standards, and robust handling of missing values or edge cases.
"""
import io
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend safe for Streamlit and server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

# Consistent dashboard color scheme
COLOR_SCHEME = {
    'primary': '#00b4d8',
    'secondary': '#0077b6',
    'accent': '#90e0ef',
    'warning': '#f4a261',
    'danger': '#e76f51',
    'success': '#2a9d8f',
    'background': '#0e1117',
    'card_bg': '#1a1f2e',
    'text': '#fafafa'
}

# AQI category colors (following Indian National Air Quality Index standards)
# Good (0-50): Green
# Satisfactory (51-100): Light Green
# Moderate (101-200): Yellow
# Poor (201-300): Orange
# Very Poor (301-400): Red
# Severe (401-500): Dark Red / Maroon
AQI_COLORS = {
    'Good': '#009966',
    'Satisfactory': '#58a64c',
    'Moderate': '#ffde33',
    'Poor': '#ff9933',
    'Very Poor': '#cc0033',
    'Severe': '#7e0023'
}

# City color palette for visual distinction across multiple lines and charts
CITY_COLORS = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel


def create_plotly_layout(title: str, xaxis_title: str = '', yaxis_title: str = '',
                         height: int = 500) -> dict:
    """
    Create a consistent dark-theme Plotly layout dictionary.
    
    Parameters:
    -----------
    title : str
        Main title for the chart.
    xaxis_title : str
        Label for the X-axis.
    yaxis_title : str
        Label for the Y-axis.
    height : int
        Figure height in pixels.

    Returns:
    --------
    dict: Layout configuration dictionary for plotly figures.
    """
    return dict(
        title=dict(text=title, font=dict(size=18, color=COLOR_SCHEME['text'])),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template='plotly_dark',
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLOR_SCHEME['text']),
        legend=dict(bgcolor='rgba(0,0,0,0.3)'),
        margin=dict(l=60, r=30, t=60, b=60)
    )


def plot_trend_line(df: pd.DataFrame, cities: List[str], pollutant: str,
                    freq: str = 'M') -> go.Figure:
    """
    Create an interactive line chart showing pollution trends over time.
    Supports multiple cities for comparison.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'Date', 'City', and the pollutant column.
    cities : List[str]
        List of city names to include in the trend line chart.
    pollutant : str
        The pollutant column name (e.g., 'PM2.5', 'NO2', 'AQI').
    freq : str
        Resampling frequency ('D'=Daily, 'W'=Weekly, 'M'=Monthly, 'Q'=Quarterly, 'Y'=Yearly).

    Returns:
    --------
    go.Figure: Interactive line chart.
    """
    fig = go.Figure()
    
    freq_labels = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly', 'Y': 'Yearly'}
    freq_label = freq_labels.get(freq, freq)
    
    if df.empty or pollutant not in df.columns:
        layout = create_plotly_layout(
            title=f'{freq_label} {pollutant} Trends (No Data Available)',
            xaxis_title='Date',
            yaxis_title=f'{pollutant} Concentration',
            height=500
        )
        fig.update_layout(**layout)
        return fig

    # Map pandas resample frequencies safely (e.g. Pandas 2.2 prefers 'ME' over 'M')
    resample_freq = freq
    
    for i, city in enumerate(cities):
        city_data = df[df['City'] == city].copy() if 'City' in df.columns else df.copy()
        
        if city_data.empty or pollutant not in city_data.columns:
            continue
        
        # Ensure Date column is in datetime format
        if 'Date' in city_data.columns:
            city_data['Date'] = pd.to_datetime(city_data['Date'])
            try:
                agg_data = city_data.set_index('Date')[pollutant].resample(resample_freq).mean().reset_index()
            except ValueError:
                # Fallback for pandas versions with ME instead of M
                fallback_freq = 'ME' if resample_freq == 'M' else 'QE' if resample_freq == 'Q' else 'YE' if resample_freq == 'Y' else resample_freq
                agg_data = city_data.set_index('Date')[pollutant].resample(fallback_freq).mean().reset_index()
                
            agg_data = agg_data.dropna()
        else:
            continue
        
        if agg_data.empty:
            continue

        color = CITY_COLORS[i % len(CITY_COLORS)]
        
        fig.add_trace(go.Scatter(
            x=agg_data['Date'],
            y=agg_data[pollutant],
            name=city,
            mode='lines+markers',
            line=dict(width=2, color=color),
            marker=dict(size=4),
            hovertemplate=f'<b>{city}</b><br>Date: %{{x}}<br>{pollutant}: %{{y:.1f}}<extra></extra>'
        ))
    
    layout = create_plotly_layout(
        title=f'{freq_label} {pollutant} Trends',
        xaxis_title='Date',
        yaxis_title=f'{pollutant} Concentration',
        height=500
    )
    fig.update_layout(**layout)
    
    return fig


def plot_city_comparison_bar(df: pd.DataFrame, pollutant: str, 
                             top_n: int = 15) -> go.Figure:
    """
    Bar chart comparing average pollutant levels across cities.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'City' and pollutant column.
    pollutant : str
        Pollutant name to compare.
    top_n : int
        Maximum number of cities to display.

    Returns:
    --------
    go.Figure: Horizontal bar chart sorted by concentration.
    """
    if 'City' not in df.columns or pollutant not in df.columns or df.empty:
        return go.Figure()
    
    city_means = df.groupby('City')[pollutant].mean().dropna().sort_values(ascending=True)
    
    if city_means.empty:
        return go.Figure()

    if len(city_means) > top_n:
        city_means = city_means.tail(top_n)
    
    # Color gradient based on value (higher = more red/warning)
    colors = []
    max_val = city_means.max()
    min_val = city_means.min()
    for val in city_means.values:
        if max_val == min_val:
            ratio = 0.5
        else:
            ratio = (val - min_val) / (max_val - min_val)
        # Green to Yellow to Red gradient
        if ratio < 0.5:
            r = int(255 * (ratio * 2))
            g = 200
        else:
            r = 255
            g = int(200 * (1 - (ratio - 0.5) * 2))
        colors.append(f'rgb({r},{g},50)')
    
    fig = go.Figure(go.Bar(
        x=city_means.values,
        y=city_means.index,
        orientation='h',
        marker_color=colors,
        hovertemplate='<b>%{y}</b><br>Mean ' + pollutant + ': %{x:.1f}<extra></extra>'
    ))
    
    layout = create_plotly_layout(
        title=f'Average {pollutant} by City (in selected data)',
        xaxis_title=f'Mean {pollutant}',
        yaxis_title='',
        height=max(400, len(city_means) * 35 + 100)
    )
    fig.update_layout(**layout)
    
    return fig


def plot_aqi_ranking(df: pd.DataFrame, top_n: int = 15) -> Optional[go.Figure]:
    """
    Bar chart ranking cities by average AQI with standard Indian AQI colors.
    Returns None if AQI data is not available.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'City' and 'AQI' columns.
    top_n : int
        Number of top ranked cities to show.

    Returns:
    --------
    Optional[go.Figure]: Plotly horizontal bar figure or None.
    """
    if 'AQI' not in df.columns or 'City' not in df.columns or df.empty:
        return None
    
    city_aqi = df.groupby('City')['AQI'].mean().dropna().sort_values(ascending=True)
    
    if city_aqi.empty:
        return None
    
    if len(city_aqi) > top_n:
        city_aqi = city_aqi.tail(top_n)
    
    # AQI-based coloring following official guidelines
    colors = []
    for val in city_aqi.values:
        if val <= 50:
            colors.append(AQI_COLORS['Good'])
        elif val <= 100:
            colors.append(AQI_COLORS['Satisfactory'])
        elif val <= 200:
            colors.append(AQI_COLORS['Moderate'])
        elif val <= 300:
            colors.append(AQI_COLORS['Poor'])
        elif val <= 400:
            colors.append(AQI_COLORS['Very Poor'])
        else:
            colors.append(AQI_COLORS['Severe'])
    
    fig = go.Figure(go.Bar(
        x=city_aqi.values,
        y=city_aqi.index,
        orientation='h',
        marker_color=colors,
        hovertemplate='<b>%{y}</b><br>Mean AQI: %{x:.1f}<extra></extra>'
    ))
    
    layout = create_plotly_layout(
        title='Average AQI by City (within selected dataset)',
        xaxis_title='Mean AQI',
        yaxis_title='',
        height=max(400, len(city_aqi) * 35 + 100)
    )
    fig.update_layout(**layout)
    
    return fig


def plot_monthly_trend(monthly_stats: pd.DataFrame, pollutant: str) -> go.Figure:
    """
    Line + bar chart showing monthly pollution patterns (Mean & Median).
    
    Parameters:
    -----------
    monthly_stats : pd.DataFrame
        DataFrame containing 'Month_Name', 'mean', and 'median' columns.
    pollutant : str
        Pollutant name displayed in title and axes.

    Returns:
    --------
    go.Figure: Combo bar and line chart.
    """
    if monthly_stats.empty:
        return go.Figure()

    fig = make_subplots(specs=[[{'secondary_y': True}]])
    
    fig.add_trace(
        go.Bar(
            x=monthly_stats['Month_Name'],
            y=monthly_stats['mean'],
            name=f'Mean {pollutant}',
            marker_color=COLOR_SCHEME['primary'],
            opacity=0.7,
            hovertemplate='Month: %{x}<br>Mean: %{y:.1f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_stats['Month_Name'],
            y=monthly_stats['median'],
            name=f'Median {pollutant}',
            mode='lines+markers',
            line=dict(color=COLOR_SCHEME['warning'], width=3),
            hovertemplate='Month: %{x}<br>Median: %{y:.1f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    layout = create_plotly_layout(
        title=f'Monthly {pollutant} Pattern',
        xaxis_title='Month',
        yaxis_title=f'{pollutant} Concentration',
        height=450
    )
    fig.update_layout(**layout)
    
    return fig


def plot_seasonal_comparison(seasonal_stats: pd.DataFrame, pollutant: str) -> go.Figure:
    """
    Bar chart comparing pollution across Indian seasons (Winter, Summer, Monsoon, Post-Monsoon).
    
    Parameters:
    -----------
    seasonal_stats : pd.DataFrame
        DataFrame with 'Season', 'mean', 'std', and 'count' columns.
    pollutant : str
        Pollutant name.

    Returns:
    --------
    go.Figure: Bar chart with error bars.
    """
    if seasonal_stats.empty:
        return go.Figure()

    colors = {
        'Winter': '#4cc9f0',
        'Summer': '#f4a261',
        'Monsoon': '#2a9d8f',
        'Post-Monsoon': '#e76f51'
    }
    
    bar_colors = [colors.get(s, COLOR_SCHEME['primary']) for s in seasonal_stats['Season']]
    
    fig = go.Figure(go.Bar(
        x=seasonal_stats['Season'],
        y=seasonal_stats['mean'],
        marker_color=bar_colors,
        error_y=dict(
            type='data',
            array=seasonal_stats['std'].fillna(0).values,
            visible=True,
            color=COLOR_SCHEME['text'],
            thickness=1.5
        ),
        hovertemplate='<b>%{x}</b><br>Mean: %{y:.1f}<br>Count: ' + 
                      seasonal_stats['count'].astype(str).values + '<extra></extra>'
    ))
    
    layout = create_plotly_layout(
        title=f'Seasonal {pollutant} Comparison',
        xaxis_title='Season',
        yaxis_title=f'Mean {pollutant}',
        height=450
    )
    fig.update_layout(**layout)
    
    return fig


def plot_heatmap_city_month(pivot_df: pd.DataFrame, pollutant: str) -> go.Figure:
    """
    Heatmap showing city x month pollution patterns.
    
    Parameters:
    -----------
    pivot_df : pd.DataFrame
        2D matrix with cities as index and months as columns.
    pollutant : str
        Pollutant name.

    Returns:
    --------
    go.Figure: Plotly Heatmap.
    """
    if pivot_df.empty:
        return go.Figure()

    fig = go.Figure(go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='YlOrRd',
        colorbar=dict(title=pollutant),
        hovertemplate='City: %{y}<br>Month: %{x}<br>' + pollutant + ': %{z:.1f}<extra></extra>'
    ))
    
    layout = create_plotly_layout(
        title=f'{pollutant} Levels: City × Month Heatmap',
        xaxis_title='Month',
        yaxis_title='City',
        height=max(400, len(pivot_df) * 30 + 150)
    )
    fig.update_layout(**layout)
    
    return fig


def plot_distribution(df: pd.DataFrame, pollutant: str) -> go.Figure:
    """
    Histogram with box plot for pollutant distribution analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the pollutant column.
    pollutant : str
        Name of pollutant column.

    Returns:
    --------
    go.Figure: Subplot with histogram and box plot.
    """
    if pollutant not in df.columns:
        return go.Figure()

    values = df[pollutant].dropna()
    
    if values.empty:
        return go.Figure()

    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                        shared_xaxes=True, vertical_spacing=0.05)
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=50,
            marker_color=COLOR_SCHEME['primary'],
            opacity=0.7,
            name='Distribution',
            hovertemplate=f'{pollutant}: %{{x:.1f}}<br>Count: %{{y}}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(
            x=values,
            name='Box Plot',
            marker_color=COLOR_SCHEME['accent'],
            line_color=COLOR_SCHEME['primary']
        ),
        row=2, col=1
    )
    
    layout = create_plotly_layout(
        title=f'{pollutant} Distribution',
        xaxis_title=f'{pollutant} Concentration',
        height=500
    )
    fig.update_layout(**layout, showlegend=False)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    
    return fig


def plot_box_by_city(df: pd.DataFrame, pollutant: str, 
                     cities: Optional[List[str]] = None) -> go.Figure:
    """
    Box plots comparing pollutant distribution across cities.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'City' and pollutant columns.
    pollutant : str
        Target pollutant column.
    cities : Optional[List[str]]
        Optional subset of cities to include.

    Returns:
    --------
    go.Figure: Box plot figure across cities.
    """
    if 'City' not in df.columns or pollutant not in df.columns or df.empty:
        return go.Figure()

    data = df.copy()
    if cities:
        data = data[data['City'].isin(cities)]
    
    if data.empty:
        return go.Figure()

    # Sort cities by median value for cleaner visualization
    city_medians = data.groupby('City')[pollutant].median().dropna().sort_values(ascending=False)
    city_order = city_medians.index.tolist()
    
    fig = go.Figure()
    
    for i, city in enumerate(city_order):
        city_data = data[data['City'] == city][pollutant].dropna()
        if city_data.empty:
            continue
        fig.add_trace(go.Box(
            y=city_data,
            name=city,
            marker_color=CITY_COLORS[i % len(CITY_COLORS)],
            boxpoints='outliers'
        ))
    
    layout = create_plotly_layout(
        title=f'{pollutant} Distribution by City',
        yaxis_title=f'{pollutant} Concentration',
        height=500
    )
    fig.update_layout(**layout)
    
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> plt.Figure:
    """
    Create a correlation heatmap using Seaborn (static plot).
    Uses Matplotlib and Seaborn for a publication-quality heatmap.
    
    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        Correlation matrix of pollutants.

    Returns:
    --------
    plt.Figure: Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set dark background to match dashboard theme
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    if corr_matrix.empty or len(corr_matrix) < 2:
        ax.text(0.5, 0.5, 'Insufficient data for correlation heatmap',
                horizontalalignment='center', verticalalignment='center',
                color='white', fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        return fig

    # Create heatmap with Seaborn using upper triangle mask
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.3f',
        cmap='RdYlBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1,
        linecolor='#2a2f3e',
        cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
        annot_kws={'size': 12, 'color': 'white'},
        ax=ax
    )
    
    ax.set_title('Pollutant Correlation Heatmap', fontsize=16, color='white', pad=20)
    ax.tick_params(colors='white')
    
    # Color the colorbar text
    if ax.collections and hasattr(ax.collections[0], 'colorbar') and ax.collections[0].colorbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors='white')
        cbar.set_label('Correlation Coefficient', color='white')
    
    plt.tight_layout()
    return fig


def plot_scatter_pollutants(df: pd.DataFrame, pollutant_x: str, 
                            pollutant_y: str, color_by: str = 'City') -> go.Figure:
    """
    Scatter plot between two pollutants to visualize correlation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with pollutant columns.
    pollutant_x : str
        X-axis pollutant name.
    pollutant_y : str
        Y-axis pollutant name.
    color_by : str
        Column to color points by (default: 'City').

    Returns:
    --------
    go.Figure: Scatter plot with correlation coefficient annotation.
    """
    if pollutant_x not in df.columns or pollutant_y not in df.columns or df.empty:
        return go.Figure()

    cols_to_keep = [pollutant_x, pollutant_y]
    if color_by in df.columns:
        cols_to_keep.append(color_by)

    data = df[cols_to_keep].dropna(subset=[pollutant_x, pollutant_y]).copy()
    
    if data.empty:
        return go.Figure()

    if color_by in data.columns:
        fig = px.scatter(
            data, x=pollutant_x, y=pollutant_y, color=color_by,
            opacity=0.5,
            template='plotly_dark',
            title=f'{pollutant_x} vs {pollutant_y}'
        )
    else:
        fig = px.scatter(
            data, x=pollutant_x, y=pollutant_y,
            opacity=0.5,
            template='plotly_dark',
            title=f'{pollutant_x} vs {pollutant_y}'
        )
    
    # Add trendline annotation if sufficient data points exist
    if len(data) > 2:
        corr_val = data[pollutant_x].corr(data[pollutant_y])
        if not np.isnan(corr_val):
            fig.add_annotation(
                text=f'Correlation: {corr_val:.3f}',
                xref='paper', yref='paper',
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=14, color=COLOR_SCHEME['accent']),
                bgcolor='rgba(0,0,0,0.5)'
            )
    
    layout = create_plotly_layout(
        title=f'{pollutant_x} vs {pollutant_y} (Correlation does not imply causation)',
        xaxis_title=pollutant_x,
        yaxis_title=pollutant_y,
        height=500
    )
    fig.update_layout(**layout)
    
    return fig


def plot_aqi_distribution_pie(df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Pie chart showing AQI category distribution.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'AQI_Bucket' column.

    Returns:
    --------
    Optional[go.Figure]: Pie chart or None if no data is available.
    """
    if 'AQI_Bucket' not in df.columns or df.empty:
        return None
    
    distribution = df['AQI_Bucket'].dropna().value_counts()
    
    if distribution.empty:
        return None
    
    colors = [AQI_COLORS.get(cat, '#666666') for cat in distribution.index]
    
    fig = go.Figure(go.Pie(
        labels=distribution.index,
        values=distribution.values,
        marker=dict(colors=colors),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>',
        textinfo='percent+label'
    ))
    
    layout = create_plotly_layout(
        title='AQI Category Distribution',
        height=450
    )
    fig.update_layout(**layout)
    
    return fig


def plot_missing_values(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing missing value percentages by column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.

    Returns:
    --------
    go.Figure: Horizontal bar chart of missing percentages.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text='No data available to analyze missing values.',
            xref='paper', yref='paper',
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLOR_SCHEME['warning'])
        )
        layout = create_plotly_layout(title='Missing Values Analysis', height=300)
        fig.update_layout(**layout)
        return fig

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    # Filter to only show columns with missing values
    missing_data = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values
    })
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing %', ascending=True)
    
    if missing_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text='No missing values found! Dataset is 100% complete.',
            xref='paper', yref='paper',
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color=COLOR_SCHEME['success'])
        )
        layout = create_plotly_layout(title='Missing Values Analysis', height=300)
        fig.update_layout(**layout)
        return fig
    
    colors = ['#e76f51' if pct > 20 else '#f4a261' if pct > 10 else '#2a9d8f' 
              for pct in missing_data['Missing %']]
    
    fig = go.Figure(go.Bar(
        x=missing_data['Missing %'],
        y=missing_data['Column'],
        orientation='h',
        marker_color=colors,
        text=[f"{pct:.1f}% ({cnt:,})" for pct, cnt in zip(missing_data['Missing %'], missing_data['Missing Count'])],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Missing: %{x:.1f}%<extra></extra>'
    ))
    
    layout = create_plotly_layout(
        title='Missing Values by Column',
        xaxis_title='Missing Percentage (%)',
        height=max(300, len(missing_data) * 30 + 100)
    )
    fig.update_layout(**layout)
    
    return fig


def plot_data_coverage_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Heatmap showing data coverage (completeness/observation count) by city and year.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'City' and 'Year' columns (or 'Date' from which Year can be extracted).

    Returns:
    --------
    go.Figure: Heatmap of observations per city per year.
    """
    if df.empty or 'City' not in df.columns:
        return go.Figure()
    
    data = df.copy()
    if 'Year' not in data.columns and 'Date' in data.columns:
        data['Year'] = pd.to_datetime(data['Date']).dt.year
    
    if 'Year' not in data.columns:
        return go.Figure()
    
    # Count observations per city per year
    coverage = data.groupby(['City', 'Year']).size().reset_index(name='Count')
    pivot = coverage.pivot_table(values='Count', index='City', columns='Year', fill_value=0)
    
    if pivot.empty:
        return go.Figure()

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=pivot.index,
        colorscale='Viridis',
        colorbar=dict(title='Observations'),
        hovertemplate='City: %{y}<br>Year: %{x}<br>Observations: %{z}<extra></extra>'
    ))
    
    layout = create_plotly_layout(
        title='Data Coverage: Observations by City and Year',
        xaxis_title='Year',
        yaxis_title='City',
        height=max(400, len(pivot) * 30 + 100)
    )
    fig.update_layout(**layout)
    
    return fig
