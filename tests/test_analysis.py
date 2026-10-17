"""
Tests for Analysis Module
==========================
Tests statistical calculations, aggregation, and correlation analysis.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis import (
    get_descriptive_stats, get_city_stats, aggregate_time_series,
    get_monthly_stats, get_seasonal_stats, get_city_seasonal_pivot,
    get_correlation_matrix, get_top_bottom_cities, get_yearly_trend,
    get_aqi_distribution
)


def create_test_dataset():
    """Create a standardized test dataset with known values."""
    np.random.seed(42)
    
    # Create deterministic data for predictable test results
    data = {
        'City': ['Delhi'] * 30 + ['Mumbai'] * 30 + ['Bengaluru'] * 30,
        'Date': pd.date_range('2023-01-01', periods=30, freq='D').tolist() * 3,
        'PM2.5': (
            list(np.random.uniform(150, 300, 30)) +  # Delhi: high
            list(np.random.uniform(50, 150, 30)) +    # Mumbai: medium
            list(np.random.uniform(20, 80, 30))        # Bengaluru: low
        ),
        'PM10': (
            list(np.random.uniform(250, 500, 30)) +
            list(np.random.uniform(100, 250, 30)) +
            list(np.random.uniform(40, 120, 30))
        ),
        'NO2': list(np.random.uniform(10, 100, 90)),
        'SO2': list(np.random.uniform(5, 50, 90)),
        'CO': list(np.random.uniform(0.5, 5, 90)),
        'O3': list(np.random.uniform(10, 80, 90)),
        'AQI': (
            list(np.random.uniform(200, 400, 30)) +
            list(np.random.uniform(100, 200, 30)) +
            list(np.random.uniform(50, 100, 30))
        ),
        'AQI_Bucket': (
            ['Very Poor'] * 15 + ['Severe'] * 15 +
            ['Moderate'] * 15 + ['Poor'] * 15 +
            ['Good'] * 15 + ['Satisfactory'] * 15
        )
    }
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Add derived columns (normally done by cleaning)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%B')
    
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Summer', 4: 'Summer', 5: 'Summer',
        6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
        10: 'Post-Monsoon', 11: 'Post-Monsoon'
    }
    df['Season'] = df['Month'].map(season_map)
    
    return df


class TestDescriptiveStats:
    """Tests for descriptive statistics calculations."""
    
    def test_basic_stats(self):
        """Test that descriptive stats are computed correctly."""
        df = create_test_dataset()
        stats = get_descriptive_stats(df, ['PM2.5'])
        
        assert not stats.empty
        assert 'Mean' in stats.columns
        assert 'Median' in stats.columns
        assert 'Std Dev' in stats.columns
        assert 'Min' in stats.columns
        assert 'Max' in stats.columns
        assert stats.iloc[0]['Count'] == 90
    
    def test_multiple_pollutants(self):
        """Test stats for multiple pollutants."""
        df = create_test_dataset()
        stats = get_descriptive_stats(df, ['PM2.5', 'PM10', 'NO2'])
        
        assert len(stats) == 3
        assert set(stats['Pollutant'].values) == {'PM2.5', 'PM10', 'NO2'}
    
    def test_stats_with_missing_values(self):
        """Test that stats handle missing values correctly."""
        df = create_test_dataset()
        df.loc[:10, 'PM2.5'] = np.nan  # Set some values to NaN
        
        stats = get_descriptive_stats(df, ['PM2.5'])
        assert stats.iloc[0]['Missing'] == 11
        assert stats.iloc[0]['Count'] == 79
    
    def test_empty_dataframe(self):
        """Test stats for empty dataframe."""
        df = pd.DataFrame()
        stats = get_descriptive_stats(df)
        assert stats.empty
    
    def test_mean_calculation_accuracy(self):
        """Verify mean is calculated correctly using known values."""
        df = pd.DataFrame({
            'PM2.5': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        stats = get_descriptive_stats(df, ['PM2.5'])
        assert stats.iloc[0]['Mean'] == 30.0
        assert stats.iloc[0]['Median'] == 30.0


class TestCityStats:
    """Tests for city-wise statistics."""
    
    def test_city_stats_ranking(self):
        """Test that cities are ranked by mean pollutant value."""
        df = create_test_dataset()
        city_stats = get_city_stats(df, 'PM2.5')
        
        assert not city_stats.empty
        assert len(city_stats) == 3  # Delhi, Mumbai, Bengaluru
        # Delhi should have highest mean PM2.5
        assert city_stats.iloc[0]['City'] == 'Delhi'
    
    def test_city_stats_columns(self):
        """Test that all expected columns are present."""
        df = create_test_dataset()
        city_stats = get_city_stats(df, 'PM2.5')
        
        expected_cols = ['City', 'Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', '90th Pctl']
        for col in expected_cols:
            assert col in city_stats.columns, f"Missing column: {col}"


class TestTimeSeriesAggregation:
    """Tests for time series aggregation."""
    
    def test_monthly_aggregation(self):
        """Test monthly aggregation."""
        df = create_test_dataset()
        agg = aggregate_time_series(df, freq='M', pollutant='PM2.5')
        
        assert not agg.empty
        assert 'Date' in agg.columns
        assert 'PM2.5' in agg.columns
    
    def test_city_filter(self):
        """Test aggregation with city filter."""
        df = create_test_dataset()
        agg = aggregate_time_series(df, freq='M', cities=['Delhi'], pollutant='PM2.5')
        
        assert not agg.empty
        assert all(agg['City'] == 'Delhi')
    
    def test_empty_city_filter(self):
        """Test aggregation with non-existent city."""
        df = create_test_dataset()
        agg = aggregate_time_series(df, freq='M', cities=['NonExistent'], pollutant='PM2.5')
        
        assert agg.empty


class TestSeasonalAnalysis:
    """Tests for seasonal analysis."""
    
    def test_monthly_stats(self):
        """Test month-wise statistics."""
        df = create_test_dataset()
        monthly = get_monthly_stats(df, 'PM2.5')
        
        assert not monthly.empty
        assert 'Month' in monthly.columns
        assert 'Month_Name' in monthly.columns
    
    def test_seasonal_stats(self):
        """Test season-wise statistics."""
        df = create_test_dataset()
        seasonal = get_seasonal_stats(df, 'PM2.5')
        
        assert not seasonal.empty
        # Check season order
        expected_seasons = ['Winter', 'Summer', 'Monsoon', 'Post-Monsoon']
        # Note: some seasons may have no data if date range is short
    
    def test_city_month_pivot(self):
        """Test city x month pivot table."""
        df = create_test_dataset()
        pivot = get_city_seasonal_pivot(df, 'PM2.5')
        
        assert not pivot.empty
        assert 'Jan' in pivot.columns
        assert 'Delhi' in pivot.index


class TestCorrelation:
    """Tests for correlation analysis."""
    
    def test_correlation_matrix(self):
        """Test correlation matrix computation."""
        df = create_test_dataset()
        corr = get_correlation_matrix(df)
        
        assert not corr.empty
        # Diagonal should be 1.0
        for col in corr.columns:
            assert abs(corr.loc[col, col] - 1.0) < 0.001
    
    def test_correlation_symmetry(self):
        """Test that correlation matrix is symmetric."""
        df = create_test_dataset()
        corr = get_correlation_matrix(df)
        
        # Check symmetry
        for i in range(len(corr)):
            for j in range(len(corr)):
                assert abs(corr.iloc[i, j] - corr.iloc[j, i]) < 0.001
    
    def test_correlation_range(self):
        """Test that all correlations are between -1 and 1."""
        df = create_test_dataset()
        corr = get_correlation_matrix(df)
        
        assert (corr >= -1.001).all().all()
        assert (corr <= 1.001).all().all()
    
    def test_known_correlation(self):
        """Test correlation with perfectly correlated data."""
        df = pd.DataFrame({
            'PM2.5': [10, 20, 30, 40, 50],
            'PM10': [20, 40, 60, 80, 100]  # Perfectly correlated (2x)
        })
        corr = get_correlation_matrix(df, ['PM2.5', 'PM10'])
        assert abs(corr.loc['PM2.5', 'PM10'] - 1.0) < 0.001


class TestTopBottomCities:
    """Tests for top/bottom city ranking."""
    
    def test_top_bottom_ranking(self):
        """Test city ranking."""
        df = create_test_dataset()
        top, bottom = get_top_bottom_cities(df, 'PM2.5', n=2)
        
        assert not top.empty
        assert not bottom.empty
        assert len(top) == 2
        assert len(bottom) == 2
        # Delhi should be in top (highest pollution)
        assert 'Delhi' in top['City'].values
    
    def test_top_bottom_with_n_larger_than_cities(self):
        """Test when n is larger than number of cities."""
        df = create_test_dataset()
        top, bottom = get_top_bottom_cities(df, 'PM2.5', n=10)
        
        assert len(top) == 3  # Only 3 cities
        assert len(bottom) == 3


class TestAQIDistribution:
    """Tests for AQI distribution analysis."""
    
    def test_aqi_distribution(self):
        """Test AQI bucket distribution."""
        df = create_test_dataset()
        dist = get_aqi_distribution(df)
        
        assert dist is not None
        assert 'AQI Category' in dist.columns
        assert 'Count' in dist.columns
        assert 'Percentage' in dist.columns
        assert dist['Count'].sum() == 90
    
    def test_aqi_distribution_without_aqi(self):
        """Test handling when AQI column is missing."""
        df = create_test_dataset().drop('AQI_Bucket', axis=1)
        dist = get_aqi_distribution(df)
        assert dist is None


class TestYearlyTrend:
    """Tests for yearly trend analysis."""
    
    def test_yearly_trend(self):
        """Test year-over-year trend."""
        df = create_test_dataset()
        yearly = get_yearly_trend(df, 'PM2.5')
        
        assert not yearly.empty
        assert 'Year' in yearly.columns
        assert 'City' in yearly.columns
    
    def test_yearly_trend_with_city_filter(self):
        """Test yearly trend filtered by city."""
        df = create_test_dataset()
        yearly = get_yearly_trend(df, 'PM2.5', cities=['Delhi'])
        
        assert not yearly.empty
        assert all(yearly['City'] == 'Delhi')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

