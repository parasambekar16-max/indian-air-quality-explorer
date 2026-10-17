"""
Indian Air Quality & Pollution Explorer
========================================
An interactive data science dashboard that explores air pollution
across Indian cities using real environmental data.

Built with: Streamlit, Pandas, NumPy, Plotly, Matplotlib, Seaborn

Author: Indian Air Quality Explorer Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import os

# Import project modules
from src.data_loader import load_data, load_uploaded_file, POLLUTANT_COLUMNS, NUMERIC_COLUMNS
from src.data_cleaning import clean_data, get_data_quality_report, get_zero_vs_missing_distinction
from src.analysis import (
    get_descriptive_stats, get_city_stats, aggregate_time_series,
    get_monthly_stats, get_seasonal_stats, get_city_seasonal_pivot,
    get_correlation_matrix, get_top_bottom_cities, get_yearly_trend,
    get_aqi_distribution
)
from src.visualizations import (
    plot_trend_line, plot_city_comparison_bar, plot_aqi_ranking,
    plot_monthly_trend, plot_seasonal_comparison, plot_heatmap_city_month,
    plot_distribution, plot_box_by_city, plot_correlation_heatmap,
    plot_scatter_pollutants, plot_aqi_distribution_pie, plot_missing_values,
    plot_data_coverage_heatmap, COLOR_SCHEME
)
from src.insights import generate_all_insights


# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Air Quality Explorer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS for professional styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* KPI Card Styling */
    .kpi-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #3a3f4e;
        transition: transform 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #00b4d8;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00b4d8;
        margin: 5px 0;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #00b4d8 0%, transparent 100%);
        padding: 2px 0;
        margin: 30px 0 15px 0;
        border-radius: 4px;
    }
    
    /* Insight cards */
    .insight-card {
        background: #1a1f2e;
        border-left: 4px solid #00b4d8;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.95rem;
    }
    
    /* Hide Streamlit footer */
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def render_kpi_card(label: str, value: str, icon: str = "📊"):
    """Render a styled KPI card using HTML."""
    st.markdown(f"""
    <div class="kpi-card">
        <div style="font-size: 1.5rem;">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def get_available_pollutants(df: pd.DataFrame) -> list:
    """Get list of pollutant columns available in the dataframe."""
    return [col for col in POLLUTANT_COLUMNS if col in df.columns and df[col].notna().any()]


def render_insight(text: str):
    """Render an insight card."""
    st.markdown(f'<div class="insight-card">{text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading and Initialization
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_and_clean_data(file_path=None, file_content=None, file_name=None):
    """
    Load and clean data with Streamlit caching.
    This prevents reloading/reprocessing on every interaction.
    """
    if file_content is not None:
        # Handle uploaded file
        import tempfile
        suffix = '.csv' if file_name.endswith('.csv') else '.xlsx'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        raw_df, load_meta = load_data(tmp_path)
        load_meta['source'] = 'user_upload'
        os.unlink(tmp_path)
    else:
        raw_df, load_meta = load_data(file_path)
    
    if raw_df.empty:
        return pd.DataFrame(), pd.DataFrame(), load_meta, {}
    
    cleaned_df, clean_report = clean_data(raw_df)
    return raw_df, cleaned_df, load_meta, clean_report


def init_session_state():
    """Initialize session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'uploaded_file_content' not in st.session_state:
        st.session_state.uploaded_file_content = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame):
    """Render the sidebar with navigation and filters."""
    with st.sidebar:
        st.markdown("## 🌿 Air Quality Explorer")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "📍 Navigate",
            ["🏠 Overview", "📈 Pollution Trends", "🏙️ City Comparison",
             "🍂 Seasonal Analysis", "🔬 Pollutant Analysis",
             "📋 Data Quality", "📥 Download & Explore"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Data source info
        if st.session_state.get('uploaded_file_name'):
            st.info(f"📂 Using: {st.session_state.uploaded_file_name}")
        else:
            st.info("📂 Using: Bundled Demo Dataset")
        
        # Global filters (only if data is available)
        filters = {}
        if not df.empty:
            st.markdown("### 🔧 Filters")
            
            # City filter
            if 'City' in df.columns:
                cities = sorted(df['City'].unique().tolist())
                selected_cities = st.multiselect(
                    "Select Cities",
                    options=cities,
                    default=cities,
                    help="Filter data by city/location"
                )
                filters['cities'] = selected_cities
            
            # Date range filter
            if 'Date' in df.columns and df['Date'].notna().any():
                date_min = df['Date'].min().date()
                date_max = df['Date'].max().date()
                
                date_range = st.date_input(
                    "Date Range",
                    value=(date_min, date_max),
                    min_value=date_min,
                    max_value=date_max,
                    help="Filter data by date range"
                )
                if len(date_range) == 2:
                    filters['date_start'] = pd.Timestamp(date_range[0])
                    filters['date_end'] = pd.Timestamp(date_range[1])
            
            # Pollutant selector
            available = get_available_pollutants(df)
            if available:
                filters['selected_pollutant'] = st.selectbox(
                    "Primary Pollutant",
                    options=available,
                    index=0,
                    help="Select the primary pollutant for analysis"
                )
        
        st.markdown("---")
        st.markdown(
            "<small style='color: #666;'>Built with Streamlit, Pandas, "
            "NumPy, Plotly & Seaborn</small>",
            unsafe_allow_html=True
        )
    
    return page, filters


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply sidebar filters to the dataframe."""
    if df.empty:
        return df
    
    filtered = df.copy()
    
    if 'cities' in filters and 'City' in filtered.columns:
        if filters['cities']:  # Only filter if selection is not empty
            filtered = filtered[filtered['City'].isin(filters['cities'])]
    
    if 'date_start' in filters and 'date_end' in filters and 'Date' in filtered.columns:
        filtered = filtered[
            (filtered['Date'] >= filters['date_start']) & 
            (filtered['Date'] <= filters['date_end'])
        ]
    
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Page A: Overview
# ─────────────────────────────────────────────────────────────────────────────

def page_overview(df: pd.DataFrame, filters: dict):
    """Render the Overview page with KPI cards and summary insights."""
    st.markdown("# 🏠 Dashboard Overview")
    st.markdown("A comprehensive view of the air quality data in the selected filters.")
    
    if df.empty:
        st.warning("No data available. Please check your filters or upload a dataset.")
        return
    
    # ── KPI Cards ──
    st.markdown('<div class="section-header"></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        n_cities = df['City'].nunique() if 'City' in df.columns else 0
        render_kpi_card("Cities / Locations", f"{n_cities}", "🏙️")
    
    with col2:
        render_kpi_card("Total Observations", f"{len(df):,}", "📊")
    
    with col3:
        if 'Date' in df.columns and df['Date'].notna().any():
            date_range = f"{df['Date'].min().strftime('%b %Y')} – {df['Date'].max().strftime('%b %Y')}"
        else:
            date_range = "N/A"
        render_kpi_card("Date Range", date_range, "📅")
    
    with col4:
        if 'AQI' in df.columns and df['AQI'].notna().any():
            avg_aqi = f"{df['AQI'].mean():.0f}"
        else:
            avg_aqi = "N/A"
        render_kpi_card("Avg AQI", avg_aqi, "🌡️")
    
    with col5:
        if 'AQI' in df.columns and df['AQI'].notna().any():
            max_aqi = f"{df['AQI'].max():.0f}"
        else:
            max_aqi = "N/A"
        render_kpi_card("Peak AQI", max_aqi, "⚠️")
    
    with col6:
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isna().sum().sum()
        completeness = (1 - total_missing / total_cells) * 100 if total_cells > 0 else 0
        render_kpi_card("Data Completeness", f"{completeness:.1f}%", "✅")
    
    st.markdown("")  # Spacing
    
    # ── Quick insights ──
    st.markdown("### 💡 Key Insights")
    insights = generate_all_insights(df)
    
    for category, items in insights.items():
        if items:
            for insight in items:
                render_insight(insight)
    
    # ── AQI Distribution ──
    col_left, col_right = st.columns(2)
    
    with col_left:
        aqi_pie = plot_aqi_distribution_pie(df)
        if aqi_pie:
            st.plotly_chart(aqi_pie, use_container_width=True)
        else:
            st.info("AQI category data is not available in the current dataset.")
    
    with col_right:
        # Quick pollutant summary
        st.markdown("### 📊 Pollutant Summary")
        available = get_available_pollutants(df)
        if available:
            stats = get_descriptive_stats(df, available)
            if not stats.empty:
                st.dataframe(
                    stats[['Pollutant', 'Mean', 'Median', 'Min', 'Max', 'Completeness %']],
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No pollutant data available.")


# ─────────────────────────────────────────────────────────────────────────────
# Page B: Pollution Trends
# ─────────────────────────────────────────────────────────────────────────────

def page_trends(df: pd.DataFrame, filters: dict):
    """Render the Pollution Trends page with time series charts."""
    st.markdown("# 📈 Pollution Trends Over Time")
    st.markdown("Track how pollution levels change over days, months, and years.")
    
    if df.empty:
        st.warning("No data available for trend analysis.")
        return
    
    available = get_available_pollutants(df)
    if not available:
        st.warning("No pollutant data available for trend analysis.")
        return
    
    # Controls
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Select pollutants to compare
        selected_pollutants = st.multiselect(
            "Select Pollutants",
            options=available,
            default=[available[0]] if available else [],
            help="Choose one or more pollutants to visualize"
        )
    
    with col2:
        # Select cities for comparison
        cities = sorted(df['City'].unique().tolist()) if 'City' in df.columns else []
        trend_cities = st.multiselect(
            "Compare Cities",
            options=cities,
            default=cities[:3] if len(cities) >= 3 else cities,
            help="Select cities to compare trends"
        )
    
    with col3:
        freq = st.selectbox(
            "Aggregation",
            options=['D', 'W', 'M', 'Q', 'Y'],
            format_func=lambda x: {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 
                                    'Q': 'Quarterly', 'Y': 'Yearly'}[x],
            index=2,
            help="Time aggregation level. Monthly is recommended for cleaner trends."
        )
    
    if not selected_pollutants:
        st.info("Please select at least one pollutant.")
        return
    
    if not trend_cities:
        st.info("Please select at least one city.")
        return
    
    freq_labels = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly', 'Y': 'Yearly'}
    st.caption(f"📌 Showing {freq_labels[freq]} averages. Aggregation smooths daily fluctuations.")
    
    # Trend charts for each pollutant
    for pollutant in selected_pollutants:
        fig = plot_trend_line(df, trend_cities, pollutant, freq)
        st.plotly_chart(fig, use_container_width=True)
    
    # Year-over-year comparison
    st.markdown("### 📅 Year-over-Year Comparison")
    
    pollutant_for_yearly = filters.get('selected_pollutant', available[0])
    yearly = get_yearly_trend(df, pollutant_for_yearly, trend_cities)
    
    if not yearly.empty:
        fig_yearly = px.bar(
            yearly, x='Year', y=f'Mean {pollutant_for_yearly}',
            color='City', barmode='group',
            template='plotly_dark',
            title=f'Yearly Mean {pollutant_for_yearly} by City'
        )
        fig_yearly.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450
        )
        st.plotly_chart(fig_yearly, use_container_width=True)
    else:
        st.info("Not enough data for year-over-year comparison.")


# ─────────────────────────────────────────────────────────────────────────────
# Page C: City Comparison
# ─────────────────────────────────────────────────────────────────────────────

def page_city_comparison(df: pd.DataFrame, filters: dict):
    """Render the City Comparison page."""
    st.markdown("# 🏙️ City Comparison")
    st.markdown("Compare pollution levels across Indian cities in the selected dataset.")
    
    if df.empty:
        st.warning("No data available for comparison.")
        return
    
    available = get_available_pollutants(df)
    pollutant = filters.get('selected_pollutant', available[0] if available else None)
    
    if not pollutant:
        st.warning("No pollutant data available.")
        return
    
    st.info(
        "⚠️ **Note:** Rankings are based only on the observations in the selected dataset "
        "and time range. They do not necessarily represent a nationwide ranking."
    )
    
    # Bar chart comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = plot_city_comparison_bar(df, pollutant)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        fig_aqi = plot_aqi_ranking(df)
        if fig_aqi:
            st.plotly_chart(fig_aqi, use_container_width=True)
        else:
            st.info("AQI data not available for ranking.")
    
    # Top and Bottom cities
    st.markdown("### 🏆 Highest and Lowest Recorded Levels")
    
    n_cities = min(5, df['City'].nunique()) if 'City' in df.columns else 0
    top_cities, bottom_cities = get_top_bottom_cities(df, pollutant, n=n_cities)
    
    col_top, col_bottom = st.columns(2)
    
    with col_top:
        st.markdown(f"**🔴 Highest Recorded Mean {pollutant}**")
        if not top_cities.empty:
            st.dataframe(top_cities, use_container_width=True, hide_index=True)
        else:
            st.write("No data available.")
    
    with col_bottom:
        st.markdown(f"**🟢 Lowest Recorded Mean {pollutant}**")
        if not bottom_cities.empty:
            st.dataframe(bottom_cities, use_container_width=True, hide_index=True)
        else:
            st.write("No data available.")
    
    # City-wise detailed statistics
    st.markdown("### 📊 City-wise Statistical Summary")
    city_stats = get_city_stats(df, pollutant)
    if not city_stats.empty:
        st.dataframe(city_stats, use_container_width=True, hide_index=True)
    
    # Box plots
    st.markdown("### 📦 Distribution Comparison")
    fig_box = plot_box_by_city(df, pollutant)
    st.plotly_chart(fig_box, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page D: Seasonal Analysis
# ─────────────────────────────────────────────────────────────────────────────

def page_seasonal(df: pd.DataFrame, filters: dict):
    """Render the Seasonal Analysis page."""
    st.markdown("# 🍂 Seasonal Analysis")
    st.markdown(
        "Explore how pollution patterns vary across seasons and months. "
        "Seasonal definitions used: **Winter** (Dec–Feb), **Summer** (Mar–May), "
        "**Monsoon** (Jun–Sep), **Post-Monsoon** (Oct–Nov)."
    )
    
    if df.empty:
        st.warning("No data available for seasonal analysis.")
        return
    
    available = get_available_pollutants(df)
    pollutant = filters.get('selected_pollutant', available[0] if available else None)
    
    if not pollutant or 'Season' not in df.columns:
        st.warning("Seasonal data not available. Date column may be missing.")
        return
    
    st.caption(
        "🌍 Note: India has diverse climatic zones. North India experiences extreme "
        "winter pollution, while southern coastal cities may show different patterns. "
        "These season definitions are generalizations and may not apply uniformly."
    )
    
    # Monthly trends
    st.markdown("### 📅 Monthly Pattern")
    
    cities_for_season = sorted(df['City'].unique().tolist()) if 'City' in df.columns else None
    season_cities = st.multiselect(
        "Select cities for seasonal analysis",
        options=cities_for_season or [],
        default=cities_for_season[:5] if cities_for_season and len(cities_for_season) >= 5 else cities_for_season,
        key="seasonal_cities"
    )
    
    monthly = get_monthly_stats(df, pollutant, season_cities if season_cities else None)
    
    if not monthly.empty:
        fig_monthly = plot_monthly_trend(monthly, pollutant)
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Seasonal comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌦️ Seasonal Comparison")
        seasonal = get_seasonal_stats(df, pollutant, season_cities if season_cities else None)
        if not seasonal.empty:
            fig_seasonal = plot_seasonal_comparison(seasonal, pollutant)
            st.plotly_chart(fig_seasonal, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Seasonal Statistics")
        if not seasonal.empty:
            st.dataframe(seasonal.rename(columns={
                'mean': 'Mean', 'median': 'Median', 'std': 'Std Dev', 'count': 'Observations'
            }), use_container_width=True, hide_index=True)
    
    # Heatmap: City × Month
    st.markdown("### 🗺️ City × Month Heatmap")
    pivot = get_city_seasonal_pivot(df, pollutant)
    if not pivot.empty:
        fig_heat = plot_heatmap_city_month(pivot, pollutant)
        st.plotly_chart(fig_heat, use_container_width=True)
    
    # Seasonal insights
    st.markdown("### 💡 Seasonal Insights")
    from src.insights import generate_seasonal_insights
    season_insights = generate_seasonal_insights(df)
    for insight in season_insights:
        render_insight(insight)


# ─────────────────────────────────────────────────────────────────────────────
# Page E: Pollutant Analysis
# ─────────────────────────────────────────────────────────────────────────────

def page_pollutant_analysis(df: pd.DataFrame, filters: dict):
    """Render the Pollutant Analysis page."""
    st.markdown("# 🔬 Pollutant Analysis")
    st.markdown("Explore pollutant distributions, relationships, and correlations.")
    
    if df.empty:
        st.warning("No data available for pollutant analysis.")
        return
    
    available = get_available_pollutants(df)
    
    if not available:
        st.warning("No pollutant columns available in the dataset.")
        return
    
    # Distribution Analysis
    st.markdown("### 📊 Pollutant Distributions")
    
    dist_pollutant = st.selectbox(
        "Select pollutant for distribution analysis",
        options=available,
        key="dist_pollutant"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_dist = plot_distribution(df, dist_pollutant)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Descriptive statistics for the selected pollutant
        stats = get_descriptive_stats(df, [dist_pollutant])
        if not stats.empty:
            st.markdown(f"**{dist_pollutant} Statistics**")
            # Transpose for better readability
            stats_t = stats.set_index('Pollutant').T
            st.dataframe(stats_t, use_container_width=True)
    
    # Pollutant Comparison (box plots)
    st.markdown("### 📦 Multi-Pollutant Comparison")
    
    compare_pollutants = st.multiselect(
        "Select pollutants to compare",
        options=available,
        default=available[:4] if len(available) >= 4 else available,
        key="compare_pollutants"
    )
    
    if compare_pollutants:
        # Normalize values for comparison (since pollutants have different scales)
        norm_option = st.checkbox("Normalize values (z-score) for comparison", value=False,
                                   help="Normalizing allows comparing pollutants with different scales")
        
        compare_data = df[compare_pollutants].copy()
        
        if norm_option:
            # z-score normalization using NumPy
            for col in compare_pollutants:
                vals = compare_data[col].dropna()
                if len(vals) > 0:
                    mean = np.mean(vals)
                    std = np.std(vals, ddof=1)
                    if std > 0:
                        compare_data[col] = (compare_data[col] - mean) / std
        
        # Melt for box plot
        melted = compare_data.melt(var_name='Pollutant', value_name='Value')
        melted = melted.dropna()
        
        fig_compare = px.box(
            melted, x='Pollutant', y='Value',
            color='Pollutant',
            template='plotly_dark',
            title='Pollutant Distribution Comparison' + (' (Normalized)' if norm_option else '')
        )
        fig_compare.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        st.plotly_chart(fig_compare, use_container_width=True)
    
    # Correlation Analysis
    st.markdown("### 🔗 Correlation Analysis")
    st.caption(
        "⚠️ **Important:** Correlation measures the strength of a linear relationship "
        "between two variables. A strong correlation does not imply that one pollutant "
        "causes the other. They may both be affected by common factors."
    )
    
    corr = get_correlation_matrix(df)
    
    if not corr.empty and len(corr) >= 2:
        col_heatmap, col_scatter = st.columns([1, 1])
        
        with col_heatmap:
            # Seaborn correlation heatmap (static)
            fig_corr = plot_correlation_heatmap(corr)
            st.pyplot(fig_corr)
        
        with col_scatter:
            st.markdown("**Scatter Plot Explorer**")
            scatter_x = st.selectbox("X-axis pollutant", options=available, index=0, key="scatter_x")
            scatter_y_options = [p for p in available if p != scatter_x]
            if scatter_y_options:
                scatter_y = st.selectbox("Y-axis pollutant", options=scatter_y_options, index=0, key="scatter_y")
                
                fig_scatter = plot_scatter_pollutants(df, scatter_x, scatter_y)
                st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Not enough numeric pollutant columns for correlation analysis.")
    
    # Correlation insights
    st.markdown("### 💡 Correlation Insights")
    from src.insights import generate_correlation_insights
    corr_insights = generate_correlation_insights(df)
    for insight in corr_insights:
        render_insight(insight)


# ─────────────────────────────────────────────────────────────────────────────
# Page F: Data Quality
# ─────────────────────────────────────────────────────────────────────────────

def page_data_quality(df: pd.DataFrame, raw_df: pd.DataFrame, clean_report: dict):
    """Render the Data Quality page."""
    st.markdown("# 📋 Data Quality Report")
    st.markdown("Understand the completeness and reliability of the dataset.")
    
    if df.empty:
        st.warning("No data available for quality analysis.")
        return
    
    # Cleaning Summary
    if clean_report:
        st.markdown("### 🧹 Cleaning Summary")
        with st.expander("View cleaning steps performed", expanded=True):
            for i, step in enumerate(clean_report.get('steps', []), 1):
                st.markdown(f"{i}. {step}")
            
            st.markdown(f"- **Rows before cleaning:** {clean_report.get('rows_before', 'N/A'):,}")
            st.markdown(f"- **Rows after cleaning:** {clean_report.get('rows_after', 'N/A'):,}")
            
            removed = clean_report.get('rows_before', 0) - clean_report.get('rows_after', 0)
            if removed > 0:
                st.markdown(f"- **Rows removed:** {removed:,} (duplicates)")
    
    # Missing Values
    st.markdown("### ❓ Missing Values Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_missing = plot_missing_values(df)
        st.plotly_chart(fig_missing, use_container_width=True)
    
    with col2:
        # Missing value table
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing': df.isnull().sum().values,
            'Present': df.notnull().sum().values,
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values
        })
        missing_data = missing_data[missing_data['Missing'] > 0].sort_values('Missing %', ascending=False)
        
        if not missing_data.empty:
            st.dataframe(missing_data, use_container_width=True, hide_index=True)
        else:
            st.success("No missing values found!")
    
    # Zero vs Missing distinction
    st.markdown("### 0️⃣ Zero Readings vs Missing Values")
    st.caption(
        "A zero reading may indicate a very low (below detection limit) concentration, "
        "while a missing value means no measurement was recorded."
    )
    
    zero_missing = get_zero_vs_missing_distinction(df)
    if zero_missing:
        zm_df = pd.DataFrame([
            {'Pollutant': k, 'Zero Readings': v['zero_readings'], 
             'Missing Values': v['missing_values']}
            for k, v in zero_missing.items()
        ])
        st.dataframe(zm_df, use_container_width=True, hide_index=True)
    
    # Duplicate rows
    st.markdown("### 🔄 Duplicate Rows")
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        st.warning(f"Found {dup_count} duplicate rows in the current data.")
    else:
        st.success("No duplicate rows found.")
    
    # Data Coverage
    st.markdown("### 🗺️ Data Coverage")
    fig_coverage = plot_data_coverage_heatmap(df)
    st.plotly_chart(fig_coverage, use_container_width=True)
    
    # Numeric Column Summary
    st.markdown("### 📊 Numeric Column Summary")
    quality_report = get_data_quality_report(df)
    
    if 'numeric_stats' in quality_report:
        numeric_df = pd.DataFrame(quality_report['numeric_stats']).T
        numeric_df.index.name = 'Pollutant'
        st.dataframe(numeric_df.reset_index(), use_container_width=True, hide_index=True)
    
    # Downloadable quality report
    st.markdown("### 📥 Download Quality Report")
    report_text = generate_quality_report_text(df, clean_report, quality_report)
    
    st.download_button(
        label="📥 Download Data Quality Report (.txt)",
        data=report_text,
        file_name="data_quality_report.txt",
        mime="text/plain"
    )


def generate_quality_report_text(df, clean_report, quality_report):
    """Generate a text-based data quality report for download."""
    lines = []
    lines.append("=" * 60)
    lines.append("INDIAN AIR QUALITY DATA - QUALITY REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    lines.append(f"Total Rows: {len(df):,}")
    lines.append(f"Total Columns: {len(df.columns)}")
    lines.append(f"Overall Completeness: {quality_report.get('overall_completeness', 'N/A')}%")
    lines.append("")
    
    if 'date_range' in quality_report:
        dr = quality_report['date_range']
        lines.append(f"Date Range: {dr['start']} to {dr['end']} ({dr['span_days']} days)")
    
    lines.append("")
    lines.append("CLEANING STEPS:")
    lines.append("-" * 40)
    for step in clean_report.get('steps', []):
        lines.append(f"  - {step}")
    
    lines.append("")
    lines.append("MISSING VALUES:")
    lines.append("-" * 40)
    for col, info in quality_report.get('missing_values', {}).items():
        if info['count'] > 0:
            lines.append(f"  {col}: {info['count']} ({info['percentage']}%)")
    
    lines.append("")
    lines.append("NUMERIC STATISTICS:")
    lines.append("-" * 40)
    for col, stats in quality_report.get('numeric_stats', {}).items():
        lines.append(f"  {col}:")
        lines.append(f"    Mean={stats['mean']}, Median={stats['median']}, "
                      f"Std={stats['std']}, Min={stats['min']}, Max={stats['max']}")
        lines.append(f"    Missing={stats['missing']} ({stats['missing_pct']}%)")
    
    lines.append("")
    lines.append(f"Duplicate Rows: {quality_report.get('duplicate_rows', 0)}")
    lines.append("")
    lines.append("Report generated by Indian Air Quality & Pollution Explorer")
    
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Page G: Download & Explore
# ─────────────────────────────────────────────────────────────────────────────

def page_download(df: pd.DataFrame, raw_df: pd.DataFrame, filters: dict):
    """Render the Download & Explore page."""
    st.markdown("# 📥 Download & Explore")
    st.markdown("Upload your own data, preview datasets, and download processed results.")
    
    # File Upload Section
    st.markdown("### 📤 Upload Your Own CSV")
    st.caption(
        "Upload a CSV file with columns: City/Location, Date, and at least one pollutant "
        "(PM2.5, PM10, NO2, SO2, CO, O3). AQI column is optional."
    )
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your air quality dataset"
    )
    
    if uploaded_file is not None:
        # Store in session state and trigger reload
        file_content = uploaded_file.read()
        st.session_state.uploaded_file_content = file_content
        st.session_state.uploaded_file_name = uploaded_file.name
        st.success(f"✅ File '{uploaded_file.name}' uploaded! Please reload the page to analyze it.")
        st.button("🔄 Reload with uploaded data", on_click=lambda: st.cache_data.clear())
    
    st.markdown("---")
    
    # Preview Raw Data
    st.markdown("### 📋 Original Dataset Preview")
    if not raw_df.empty:
        st.dataframe(raw_df.head(100), use_container_width=True, height=300)
        st.caption(f"Showing first 100 of {len(raw_df):,} rows.")
    else:
        st.info("No original data available.")
    
    # Preview Processed Data
    st.markdown("### 🧹 Processed Dataset Preview")
    if not df.empty:
        st.dataframe(df.head(100), use_container_width=True, height=300)
        st.caption(f"Showing first 100 of {len(df):,} rows.")
    else:
        st.info("No processed data available.")
    
    st.markdown("---")
    
    # Download Section
    st.markdown("### 📥 Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not df.empty:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Cleaned Dataset (CSV)",
                data=csv_data,
                file_name="cleaned_air_quality.csv",
                mime="text/csv"
            )
    
    with col2:
        if not df.empty:
            # Aggregated summary
            available = get_available_pollutants(df)
            if available:
                summary_stats = get_descriptive_stats(df, available)
                if not summary_stats.empty:
                    csv_summary = summary_stats.to_csv(index=False)
                    st.download_button(
                        label="📥 Aggregated Summary (CSV)",
                        data=csv_summary,
                        file_name="aggregated_summary.csv",
                        mime="text/csv"
                    )
    
    with col3:
        if not df.empty:
            quality_report = get_data_quality_report(df)
            report_text = generate_quality_report_text(df, {}, quality_report)
            st.download_button(
                label="📥 Quality Report (TXT)",
                data=report_text,
                file_name="data_quality_report.txt",
                mime="text/plain"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main application entry point."""
    init_session_state()
    
    # Load data
    try:
        if st.session_state.uploaded_file_content is not None:
            raw_df, cleaned_df, load_meta, clean_report = load_and_clean_data(
                file_content=st.session_state.uploaded_file_content,
                file_name=st.session_state.uploaded_file_name
            )
        else:
            raw_df, cleaned_df, load_meta, clean_report = load_and_clean_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Try uploading a dataset from the 'Download & Explore' page.")
        raw_df = pd.DataFrame()
        cleaned_df = pd.DataFrame()
        load_meta = {'errors': [str(e)]}
        clean_report = {}
    
    # Show loading errors
    if load_meta.get('errors'):
        for err in load_meta['errors']:
            st.error(f"⚠️ {err}")
    
    # Show warnings
    if load_meta.get('warnings'):
        for warn in load_meta['warnings']:
            st.warning(f"ℹ️ {warn}")
    
    # Sidebar navigation and filters
    page, filters = render_sidebar(cleaned_df)
    
    # Apply filters
    filtered_df = apply_filters(cleaned_df, filters)
    
    # Handle empty filtered results
    if filtered_df.empty and not cleaned_df.empty:
        st.warning(
            "No data matches the current filter selection. "
            "Try adjusting the date range or city selection in the sidebar."
        )
    
    # Route to selected page
    if page == "🏠 Overview":
        page_overview(filtered_df, filters)
    elif page == "📈 Pollution Trends":
        page_trends(filtered_df, filters)
    elif page == "🏙️ City Comparison":
        page_city_comparison(filtered_df, filters)
    elif page == "🍂 Seasonal Analysis":
        page_seasonal(filtered_df, filters)
    elif page == "🔬 Pollutant Analysis":
        page_pollutant_analysis(filtered_df, filters)
    elif page == "📋 Data Quality":
        page_data_quality(filtered_df, raw_df, clean_report)
    elif page == "📥 Download & Explore":
        page_download(filtered_df, raw_df, filters)


if __name__ == "__main__":
    main()

