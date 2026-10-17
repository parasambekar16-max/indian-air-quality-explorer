"""
Tests for Data Cleaning Module
===============================
Tests data cleaning, validation, and preprocessing operations.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_cleaning import clean_data, get_data_quality_report, get_zero_vs_missing_distinction
from src.data_loader import load_data, normalize_columns, validate_dataset


class TestDataLoading:
    """Tests for data loading functionality."""
    
    def test_load_default_dataset(self):
        """Test that the bundled demo dataset loads successfully."""
        df, meta = load_data()
        assert not df.empty, "Default dataset should not be empty"
        assert meta['source'] == 'bundled_demo'
        assert meta['original_rows'] > 0
    
    def test_load_nonexistent_file(self):
        """Test graceful handling of missing file."""
        df, meta = load_data("nonexistent_file.csv")
        assert df.empty
        assert len(meta['errors']) > 0
    
    def test_column_normalization(self):
        """Test that column name aliases are mapped correctly."""
        test_df = pd.DataFrame({
            'city': ['Delhi'],
            'date': ['2023-01-01'],
            'pm2.5': [100.0],
            'pm10': [200.0]
        })
        normalized = normalize_columns(test_df)
        assert 'City' in normalized.columns
        assert 'Date' in normalized.columns
        assert 'PM2.5' in normalized.columns
        assert 'PM10' in normalized.columns
    
    def test_validate_complete_dataset(self):
        """Test validation of a complete dataset."""
        df = pd.DataFrame({
            'City': ['Delhi', 'Mumbai'],
            'Date': ['2023-01-01', '2023-01-02'],
            'PM2.5': [100.0, 50.0]
        })
        is_valid, found, missing = validate_dataset(df)
        assert is_valid
        assert len(missing) == 0
    
    def test_validate_missing_city(self):
        """Test validation fails when City column is missing."""
        df = pd.DataFrame({
            'Date': ['2023-01-01'],
            'PM2.5': [100.0]
        })
        is_valid, found, missing = validate_dataset(df)
        assert not is_valid
        assert 'City' in missing
    
    def test_validate_missing_pollutants(self):
        """Test validation fails when no pollutant columns are present."""
        df = pd.DataFrame({
            'City': ['Delhi'],
            'Date': ['2023-01-01'],
            'SomeOtherColumn': [42]
        })
        is_valid, found, missing = validate_dataset(df)
        assert not is_valid


class TestDataCleaning:
    """Tests for data cleaning operations."""
    
    def get_sample_data(self):
        """Create a sample dataset for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'City': np.random.choice(['Delhi', 'Mumbai', 'Bengaluru'], 100),
            'Date': [d.strftime('%Y-%m-%d') for d in dates],
            'PM2.5': np.random.uniform(20, 300, 100),
            'PM10': np.random.uniform(40, 500, 100),
            'NO2': np.random.uniform(10, 100, 100),
            'SO2': np.random.uniform(5, 50, 100),
            'CO': np.random.uniform(0.5, 5, 100),
            'O3': np.random.uniform(10, 80, 100),
            'AQI': np.random.uniform(50, 400, 100),
            'AQI_Bucket': np.random.choice(
                ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor'], 100
            )
        })
    
    def test_clean_basic(self):
        """Test basic cleaning on a normal dataset."""
        df = self.get_sample_data()
        cleaned, report = clean_data(df)
        
        assert not cleaned.empty
        assert report['rows_before'] == 100
        assert report['rows_after'] > 0
        assert 'Date' in cleaned.columns
        assert pd.api.types.is_datetime64_any_dtype(cleaned['Date'])
    
    def test_derived_columns_added(self):
        """Test that Year, Month, Season columns are added."""
        df = self.get_sample_data()
        cleaned, report = clean_data(df)
        
        assert 'Year' in cleaned.columns
        assert 'Month' in cleaned.columns
        assert 'Season' in cleaned.columns
        assert 'Month_Name' in cleaned.columns
        assert 'Day_of_Week' in cleaned.columns
    
    def test_season_mapping(self):
        """Test that seasons are mapped correctly for India."""
        df = pd.DataFrame({
            'City': ['Delhi'] * 12,
            'Date': [f'2023-{m:02d}-15' for m in range(1, 13)],
            'PM2.5': [100] * 12
        })
        cleaned, _ = clean_data(df)
        
        # January should be Winter
        jan = cleaned[cleaned['Month'] == 1]['Season'].iloc[0]
        assert jan == 'Winter'
        
        # April should be Summer
        apr = cleaned[cleaned['Month'] == 4]['Season'].iloc[0]
        assert apr == 'Summer'
        
        # July should be Monsoon
        jul = cleaned[cleaned['Month'] == 7]['Season'].iloc[0]
        assert jul == 'Monsoon'
        
        # October should be Post-Monsoon
        oct_val = cleaned[cleaned['Month'] == 10]['Season'].iloc[0]
        assert oct_val == 'Post-Monsoon'
    
    def test_duplicate_removal(self):
        """Test that duplicates are detected and removed."""
        df = pd.DataFrame({
            'City': ['Delhi', 'Mumbai'],
            'Date': ['2023-01-01', '2023-01-02'],
            'PM2.5': [100, 50]
        })
        # Add exact duplicate of row 0
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        
        cleaned, report = clean_data(df)
        assert report['duplicates_found'] == 1
        assert report['duplicates_removed'] == 1
        assert len(cleaned) == 2
    
    def test_negative_values_removed(self):
        """Test that negative pollutant values are set to NaN."""
        df = pd.DataFrame({
            'City': ['Delhi', 'Mumbai'],
            'Date': ['2023-01-01', '2023-01-02'],
            'PM2.5': [-10, 100],
            'PM10': [200, -5]
        })
        cleaned, report = clean_data(df)
        
        # Negative values should be NaN now
        assert pd.isna(cleaned.iloc[0]['PM2.5'])
        assert pd.isna(cleaned.iloc[1]['PM10'])  # Note: order may change after sort
    
    def test_empty_dataframe(self):
        """Test cleaning of empty dataframe."""
        df = pd.DataFrame()
        cleaned, report = clean_data(df)
        assert cleaned.empty
        assert report['rows_before'] == 0
    
    def test_invalid_dates_handled(self):
        """Test handling of invalid date strings."""
        df = pd.DataFrame({
            'City': ['Delhi', 'Mumbai', 'Bengaluru'],
            'Date': ['2023-01-01', 'not-a-date', '2023-03-15'],
            'PM2.5': [100, 200, 150]
        })
        cleaned, report = clean_data(df)
        assert report['invalid_dates'] == 1
    
    def test_non_numeric_pollutant_conversion(self):
        """Test conversion of non-numeric pollutant values."""
        df = pd.DataFrame({
            'City': ['Delhi', 'Mumbai'],
            'Date': ['2023-01-01', '2023-01-02'],
            'PM2.5': ['100.5', 'invalid']
        })
        cleaned, report = clean_data(df)
        assert 'PM2.5' in report.get('non_numeric_converted', {})
    
    def test_high_values_retained(self):
        """Test that legitimately high pollution values are NOT removed."""
        df = pd.DataFrame({
            'City': ['Delhi', 'Mumbai'],
            'Date': ['2023-01-01', '2023-01-02'],
            'PM2.5': [450, 100]  # 450 is extreme but real in Delhi
        })
        cleaned, report = clean_data(df)
        assert 450 in cleaned['PM2.5'].values


class TestDataQualityReport:
    """Tests for data quality reporting."""
    
    def test_quality_report_structure(self):
        """Test the structure of a quality report."""
        df = pd.DataFrame({
            'City': ['Delhi', 'Mumbai', 'Delhi'],
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'PM2.5': [100, np.nan, 150],
            'PM10': [200, 300, 250]
        })
        report = get_data_quality_report(df)
        
        assert 'total_rows' in report
        assert 'total_columns' in report
        assert 'missing_values' in report
        assert 'overall_completeness' in report
        assert 'duplicate_rows' in report
        assert report['total_rows'] == 3
    
    def test_missing_value_calculation(self):
        """Test that missing values are calculated correctly."""
        df = pd.DataFrame({
            'City': ['Delhi', 'Mumbai'],
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'PM2.5': [100, np.nan]
        })
        report = get_data_quality_report(df)
        
        pm25_missing = report['missing_values']['PM2.5']
        assert pm25_missing['count'] == 1
        assert pm25_missing['percentage'] == 50.0
    
    def test_empty_dataset_report(self):
        """Test quality report for empty dataset."""
        df = pd.DataFrame()
        report = get_data_quality_report(df)
        assert 'error' in report
    
    def test_zero_vs_missing_distinction(self):
        """Test distinction between zeros and missing values."""
        df = pd.DataFrame({
            'PM2.5': [0, 100, np.nan, 0, 50],
            'PM10': [200, np.nan, np.nan, 100, 0]
        })
        result = get_zero_vs_missing_distinction(df)
        
        assert result['PM2.5']['zero_readings'] == 2
        assert result['PM2.5']['missing_values'] == 1
        assert result['PM10']['zero_readings'] == 1
        assert result['PM10']['missing_values'] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

