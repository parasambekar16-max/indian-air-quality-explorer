# 🌿 Indian Air Quality & Pollution Explorer

An interactive data science dashboard that explores air pollution across Indian cities, built with Python, Pandas, NumPy, Plotly, Matplotlib, Seaborn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Problem Statement

Air pollution is a critical environmental and public health concern in India. With increasing urbanization and industrialization, understanding pollution patterns across Indian cities is essential. This project provides an interactive tool to explore, analyze, and visualize air quality data, making pollution patterns accessible and understandable.

## 🎯 Objectives

- **Explore** which Indian cities have higher recorded pollution levels
- **Track** how pollution levels change over time (daily, monthly, yearly)
- **Identify** which pollutants contribute to observed pollution patterns
- **Analyze** seasonal patterns and how they vary across locations
- **Discover** correlations between different pollutants
- **Assess** data completeness and reliability

> **Note**: This project focuses purely on data analysis and visualization. It does not train or require any machine learning model.

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Core programming language |
| **NumPy** | Numerical calculations: mean, std, percentiles, z-score normalization |
| **Pandas** | Data loading, cleaning, groupby aggregation, pivot tables, resampling |
| **Matplotlib** | Backend for static statistical plots (correlation heatmap) |
| **Seaborn** | Publication-quality correlation heatmaps with masked triangles |
| **Plotly** | Interactive charts: line, bar, scatter, box, heatmap, pie charts |
| **Streamlit** | Web dashboard framework with sidebar filters, KPI cards, caching |

### How Each Technology Is Used

#### NumPy
- Computing descriptive statistics: `np.mean()`, `np.median()`, `np.std()`, `np.percentile()`
- Array operations for statistical calculations
- Z-score normalization for pollutant comparison: `(x - mean) / std`
- Creating masks for correlation heatmaps: `np.triu()`

#### Pandas
- **Loading data**: `pd.read_csv()` with encoding fallback
- **Cleaning**: `pd.to_datetime()`, `pd.to_numeric()`, `.duplicated()`, `.drop_duplicates()`
- **Grouping**: `.groupby('City')['PM2.5'].agg(['mean', 'median', 'std'])`
- **Time series**: `.resample('M').mean()` for monthly aggregation
- **Pivot tables**: `.pivot_table(values='PM2.5', index='City', columns='Month')`
- **Correlation**: `.corr()` for Pearson correlation matrix
- **Missing values**: `.isna().sum()`, `.dropna()`, `.fillna()`

#### Matplotlib & Seaborn
- `seaborn.heatmap()` for correlation matrices with annotations
- Dark theme styling to match the dashboard
- Masked upper triangle for cleaner correlation display

#### Plotly
- Interactive line charts with hover tooltips for pollution trends
- Horizontal bar charts with color gradients for city comparison
- Box plots for distribution analysis across cities
- Scatter plots with correlation annotations
- Heatmaps for city × month pollution patterns
- Pie charts for AQI category distribution

#### Streamlit
- `st.set_page_config()` for layout and theming
- `st.sidebar` for navigation and global filters
- `@st.cache_data` for efficient data caching
- `st.plotly_chart()` and `st.pyplot()` for chart rendering
- `st.download_button()` for CSV/report downloads
- `st.file_uploader()` for custom data upload
- `st.session_state` for maintaining upload state

## 📊 Dataset

### Bundled Demo Dataset
- **Source**: Generated demonstration data modeled after CPCB (Central Pollution Control Board) monitoring patterns
- **Original Reference**: [CPCB Air Quality Data](https://app.cpcbccr.com/ccr/#/caaqm-dashboard-all/caaqm-landing) and [data.gov.in](https://data.gov.in)
- **Cities**: 15 major Indian cities (Delhi, Mumbai, Bengaluru, Chennai, Kolkata, Hyderabad, Ahmedabad, Pune, Jaipur, Lucknow, Patna, Chandigarh, Visakhapatnam, Thiruvananthapuram, Guwahati)
- **Date Range**: January 2019 – December 2023
- **Columns**: City, Date, PM2.5, PM10, NO2, SO2, CO, O3, AQI, AQI_Bucket
- **Records**: ~27,000+ observations
- **License**: Demo data — no restrictions. Real CPCB data is under Open Government License India.

> ⚠️ **Important**: The bundled dataset is clearly labelled as **demonstration data** generated to match real-world statistical patterns. It should not be cited as real pollution measurements. Users can upload their own real datasets through the dashboard.

### Using Your Own Data
Upload any CSV file with these minimum columns:
- `City` (or `Location`, `Station`)
- `Date` (or `Datetime`)
- At least one pollutant: `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, or `O3`
- Optional: `AQI`, `AQI_Bucket`

The app automatically maps common column name variations.

## ✨ Features & Dashboard Pages

### 🏠 Page A: Overview
- KPI cards: cities count, observations, date range, average AQI, peak AQI, data completeness
- Auto-generated insights computed from current data and filters
- AQI category distribution pie chart
- Pollutant summary statistics table

### 📈 Page B: Pollution Trends
- Interactive multi-city trend line charts
- Daily/Weekly/Monthly/Quarterly/Yearly aggregation
- Multi-pollutant comparison
- Year-over-year grouped bar charts

### 🏙️ Page C: City Comparison
- Horizontal bar charts comparing average pollution by city
- AQI ranking with color-coded categories
- Top and bottom cities tables
- City-wise statistical summaries
- Box plot distribution comparison

### 🍂 Page D: Seasonal Analysis
- Monthly pollution pattern (bar + line chart)
- Seasonal comparison with error bars
- City × Month heatmap
- Season definitions: Winter (Dec–Feb), Summer (Mar–May), Monsoon (Jun–Sep), Post-Monsoon (Oct–Nov)

### 🔬 Page E: Pollutant Analysis
- Histogram + box plot distributions
- Multi-pollutant comparison with optional z-score normalization
- Seaborn correlation heatmap (lower triangle)
- Interactive scatter plots with correlation annotations
- Correlation insights (with causation disclaimer)

### 📋 Page F: Data Quality
- Missing value percentages by column (bar chart)
- Zero readings vs. missing values distinction
- Duplicate row detection
- Data coverage heatmap (city × year)
- Numeric column summary statistics
- Downloadable quality report

### 📥 Page G: Download & Explore
- CSV/Excel file upload with validation
- Original and processed data preview
- Download cleaned dataset, aggregated summaries, quality report

## 📁 Project Structure

```
indian-air-quality-explorer/
├── app.py                          # Main Streamlit application (7 pages)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── .env.example                    # Environment variable template
├── .streamlit/
│   └── config.toml                 # Streamlit theme configuration
├── data/
│   ├── sample_air_quality.csv      # Bundled demo dataset (~27K rows)
│   ├── generate_demo_data.py       # Script that generated the demo data
│   └── DATA_SOURCE.md              # Dataset source documentation
├── src/
│   ├── __init__.py                 # Package init
│   ├── data_loader.py              # Data loading & column normalization
│   ├── data_cleaning.py            # Cleaning, validation, preprocessing
│   ├── analysis.py                 # Statistical analysis & aggregation
│   ├── visualizations.py           # Plotly/Matplotlib/Seaborn charts
│   └── insights.py                 # Auto-generated data insights
└── tests/
    ├── test_data_cleaning.py       # Tests for cleaning & loading
    └── test_analysis.py            # Tests for statistical analysis
```

## 🚀 Installation & Setup (Windows)

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step-by-Step Setup

```bash
# 1. Clone or download the project
git clone https://github.com/YOUR_USERNAME/indian-air-quality-explorer.git
cd indian-air-quality-explorer

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment (Windows)
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the application
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

### Quick Start (Single Command)
```bash
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt && streamlit run app.py
```

## 📸 Screenshots

> Screenshots will be added after the first successful run.

| Page | Description |
|------|-------------|
| Overview | KPI cards, insights, AQI distribution |
| Pollution Trends | Multi-city time series with aggregation |
| City Comparison | Ranked bar charts and box plots |
| Seasonal Analysis | Monthly patterns and city × month heatmap |
| Pollutant Analysis | Correlation heatmap and scatter plots |
| Data Quality | Missing values analysis and coverage heatmap |

## 🔍 Key Findings & Limitations

### Observations (from demo data)
- Northern Indian cities (Delhi, Patna, Lucknow) consistently show higher PM2.5 and PM10 levels
- Winter months (November–February) exhibit significantly higher pollution in North India
- Strong positive correlation observed between PM2.5 and PM10
- Monsoon season generally shows lower pollutant levels due to washout effect
- Southern coastal cities show more moderate, stable pollution patterns

### Limitations
- **Demo data**: The bundled dataset is generated demonstration data, not real measurements
- **Season definitions**: The seasons used are generalizations; India has diverse climatic zones
- **AQI calculation**: AQI values in demo data are generated, not calculated from official Indian AQI breakpoints
- **Coverage**: Real CPCB data has significant gaps for some cities and pollutants
- **Rankings**: City rankings only apply to the selected dataset and time range, not nationwide

## ✅ Tests & Verification

### Running Tests
```bash
# Activate virtual environment first
venv\Scripts\activate

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_data_cleaning.py -v
python -m pytest tests/test_analysis.py -v
```

### Test Coverage
- **Data Loading**: Default dataset loading, missing file handling, column normalization, validation
- **Data Cleaning**: Date conversion, duplicate removal, negative value handling, derived columns, season mapping
- **Analysis**: Descriptive stats accuracy, city ranking, time aggregation, correlation properties, AQI distribution
- **Edge Cases**: Empty datasets, missing columns, invalid dates, non-numeric values

## 🔮 Future Enhancements

- [ ] Integration with live CPCB API for real-time data
- [ ] Geospatial visualization on an India map
- [ ] Health impact assessment based on WHO guidelines
- [ ] Forecasting using time series models (ARIMA, Prophet)
- [ ] Multi-language support (Hindi, regional languages)
- [ ] Automated report generation as PDF
- [ ] Comparison with WHO air quality standards
- [ ] Mobile-responsive design optimization

## 📤 Publishing to GitHub

```bash
# Initialize Git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Indian Air Quality & Pollution Explorer"

# Add your remote repository
git remote add origin https://github.com/YOUR_USERNAME/indian-air-quality-explorer.git

# Push to GitHub
git push -u origin main
```

> ⚠️ Never commit API keys, passwords, or personal tokens. The `.gitignore` file is configured to exclude sensitive files.

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- Central Pollution Control Board (CPCB) for air quality monitoring standards
- data.gov.in for open data initiatives
- Streamlit community for the excellent framework
- The open-source Python data science ecosystem

---

*Built as a data science portfolio project demonstrating Python, Pandas, NumPy, and interactive visualization skills.*

