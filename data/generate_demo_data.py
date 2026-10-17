"""
Indian Air Quality Demonstration Dataset Generator
===================================================
This script generates a realistic demonstration dataset for the Indian Air Quality
& Pollution Explorer application.

Features modeled:
- 15 diverse Indian cities across Northern, Southern, Western, Eastern, and Central regions
- Daily observations from 2019-01-01 to 2023-12-31 (5 full years)
- Seasonal variations (severe winter smog in North India, monsoon washout in summer/autumn)
- Realistic pollutant correlations (PM10 roughly 1.5x - 2.0x PM2.5, NO2 traffic spikes)
- Indian CPCB NAQI breakpoints for AQI and AQI_Bucket classification
- Realistic missing data patterns (~8-12% random missing values, some missing AQI with present pollutants)
- Small percentage (~0.1%) of duplicate rows to test data cleaning pipelines
"""

import csv
import math
import os
import random
from datetime import date, timedelta

# Set random seed for reproducibility
random.seed(42)

# Directory and file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "sample_air_quality.csv")

# City profiles with baseline pollution parameters and regional characteristics
# Base PM2.5, winter multiplier, monsoon multiplier, traffic factor (NO2/CO)
CITY_PROFILES = {
    # High pollution Northern / Gangetic plains
    "Delhi": {"base_pm25": 95, "winter_mult": 2.8, "monsoon_mult": 0.45, "traffic": 1.4, "ind": 1.2},
    "Patna": {"base_pm25": 90, "winter_mult": 2.7, "monsoon_mult": 0.45, "traffic": 1.1, "ind": 1.1},
    "Lucknow": {"base_pm25": 85, "winter_mult": 2.6, "monsoon_mult": 0.48, "traffic": 1.2, "ind": 1.0},
    # Northern / Western moderate-to-high
    "Chandigarh": {"base_pm25": 55, "winter_mult": 2.2, "monsoon_mult": 0.50, "traffic": 0.9, "ind": 0.8},
    "Jaipur": {"base_pm25": 60, "winter_mult": 2.0, "monsoon_mult": 0.55, "traffic": 1.0, "ind": 0.9},
    "Ahmedabad": {"base_pm25": 65, "winter_mult": 1.9, "monsoon_mult": 0.55, "traffic": 1.2, "ind": 1.4},
    # Coastal / Central / Western
    "Mumbai": {"base_pm25": 48, "winter_mult": 1.7, "monsoon_mult": 0.40, "traffic": 1.3, "ind": 1.1},
    "Kolkata": {"base_pm25": 65, "winter_mult": 2.3, "monsoon_mult": 0.42, "traffic": 1.2, "ind": 1.2},
    "Pune": {"base_pm25": 42, "winter_mult": 1.6, "monsoon_mult": 0.45, "traffic": 1.0, "ind": 0.9},
    "Hyderabad": {"base_pm25": 45, "winter_mult": 1.7, "monsoon_mult": 0.50, "traffic": 1.1, "ind": 1.0},
    # Southern / Low pollution
    "Bengaluru": {"base_pm25": 32, "winter_mult": 1.3, "monsoon_mult": 0.60, "traffic": 1.1, "ind": 0.7},
    "Chennai": {"base_pm25": 35, "winter_mult": 1.4, "monsoon_mult": 0.55, "traffic": 1.0, "ind": 0.8},
    "Thiruvananthapuram": {"base_pm25": 24, "winter_mult": 1.2, "monsoon_mult": 0.65, "traffic": 0.7, "ind": 0.5},
    "Visakhapatnam": {"base_pm25": 40, "winter_mult": 1.5, "monsoon_mult": 0.55, "traffic": 0.9, "ind": 1.3},
    # North-East
    "Guwahati": {"base_pm25": 45, "winter_mult": 2.1, "monsoon_mult": 0.40, "traffic": 0.8, "ind": 0.7},
}

START_DATE = date(2019, 1, 1)
END_DATE = date(2023, 12, 31)

def get_seasonal_factor(d: date, winter_mult: float, monsoon_mult: float) -> float:
    """
    Calculate seasonal multiplier based on month of the year.
    - Winter (Nov-Feb): Peak pollution, temperature inversions
    - Pre-monsoon/Summer (Mar-May): Dust storms, moderate dispersion
    - Monsoon (Jun-Sep): Wet precipitation washout
    - Post-monsoon (Oct): Crop residue burning in North, rising pollution
    """
    month = d.month
    day = d.day

    if month in (12, 1):
        # Peak winter
        return winter_mult * random.uniform(0.9, 1.15)
    elif month == 11:
        # Early winter / crop burning peak
        progress = day / 30.0
        return (1.0 + (winter_mult - 1.0) * (0.5 + 0.5 * progress)) * random.uniform(0.9, 1.15)
    elif month == 2:
        # Late winter
        progress = 1.0 - (day / 28.0)
        return (1.0 + (winter_mult - 1.0) * (0.4 + 0.6 * progress)) * random.uniform(0.9, 1.1)
    elif month in (3, 4, 5):
        # Summer / pre-monsoon
        return random.uniform(0.85, 1.15)
    elif month in (6, 7, 8, 9):
        # Monsoon season washout
        return monsoon_mult * random.uniform(0.85, 1.2)
    elif month == 10:
        # Post-monsoon transition
        progress = day / 31.0
        return (monsoon_mult + (1.2 - monsoon_mult) * progress) * random.uniform(0.9, 1.1)
    return 1.0


def calculate_cpcb_subindex_pm25(pm25: float) -> float:
    """Calculate CPCB AQI sub-index for PM2.5."""
    if pm25 <= 30:
        return (50.0 / 30.0) * pm25
    elif pm25 <= 60:
        return 50.0 + ((100.0 - 50.0) / (60.0 - 30.0)) * (pm25 - 30.0)
    elif pm25 <= 90:
        return 100.0 + ((200.0 - 100.0) / (90.0 - 60.0)) * (pm25 - 60.0)
    elif pm25 <= 120:
        return 200.0 + ((300.0 - 200.0) / (120.0 - 90.0)) * (pm25 - 90.0)
    elif pm25 <= 250:
        return 300.0 + ((400.0 - 300.0) / (250.0 - 120.0)) * (pm25 - 120.0)
    else:
        return 400.0 + ((500.0 - 400.0) / (380.0 - 250.0)) * min(pm25 - 250.0, 130.0)


def calculate_cpcb_subindex_pm10(pm10: float) -> float:
    """Calculate CPCB AQI sub-index for PM10."""
    if pm10 <= 50:
        return (50.0 / 50.0) * pm10
    elif pm10 <= 100:
        return 50.0 + ((100.0 - 50.0) / (100.0 - 50.0)) * (pm10 - 50.0)
    elif pm10 <= 250:
        return 100.0 + ((200.0 - 100.0) / (250.0 - 100.0)) * (pm10 - 100.0)
    elif pm10 <= 350:
        return 200.0 + ((300.0 - 200.0) / (350.0 - 250.0)) * (pm10 - 250.0)
    elif pm10 <= 430:
        return 300.0 + ((400.0 - 300.0) / (430.0 - 350.0)) * (pm10 - 350.0)
    else:
        return 400.0 + ((500.0 - 400.0) / (510.0 - 430.0)) * min(pm10 - 430.0, 80.0)


def get_aqi_bucket(aqi: float) -> str:
    """Determine Indian National AQI category bucket."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


def generate_data():
    """Generate daily air quality records and output to CSV."""
    rows = []
    current_date = START_DATE
    total_days = (END_DATE - START_DATE).days + 1

    # Generate daily records for each city
    for day_offset in range(total_days):
        d = START_DATE + timedelta(days=day_offset)
        date_str = d.strftime("%Y-%m-%d")

        # 2020 COVID lockdown dip (late March to May 2020)
        is_lockdown = (date(2020, 3, 25) <= d <= date(2020, 5, 31))
        lockdown_mult = 0.65 if is_lockdown else 1.0

        for city, profile in CITY_PROFILES.items():
            season_mult = get_seasonal_factor(d, profile["winter_mult"], profile["monsoon_mult"])
            weather_noise = random.lognormvariate(0.0, 0.2)  # Log-normal day-to-day weather variation

            # PM2.5 calculation
            pm25 = profile["base_pm25"] * season_mult * lockdown_mult * weather_noise

            # Add extreme winter spikes for Delhi / Patna / Lucknow (stubble burning / severe inversion)
            if city in ("Delhi", "Patna", "Lucknow") and d.month in (11, 12, 1):
                if random.random() < 0.25:
                    pm25 *= random.uniform(1.3, 1.7)
            # Clip bounds realistically
            if city in ("Bengaluru", "Chennai", "Thiruvananthapuram"):
                pm25 = max(8.0, min(pm25, 120.0))
            else:
                pm25 = max(12.0, min(pm25, 550.0))

            # PM10 correlates with PM2.5 (1.5x - 2.1x) + local dust factor
            pm10_ratio = random.uniform(1.5, 2.1)
            pm10 = pm25 * pm10_ratio + random.uniform(5.0, 25.0)

            # NO2 (10-120 range): correlated with traffic and winter inversions
            no2_base = 25.0 * profile["traffic"] * (1.2 if d.month in (11, 12, 1, 2) else 0.85)
            no2 = no2_base * random.uniform(0.7, 1.4) * (0.6 if is_lockdown else 1.0)
            no2 = max(8.0, min(no2, 125.0))

            # SO2 (5-60 range): industrial pollutant
            so2_base = 12.0 * profile["ind"] * (0.75 if d.month in (6, 7, 8, 9) else 1.0)
            so2 = so2_base * random.uniform(0.65, 1.5)
            so2 = max(4.0, min(so2, 65.0))

            # CO (0.5-8.0 mg/m3): incomplete combustion, traffic
            co_base = 1.2 * profile["traffic"] * (1.4 if d.month in (11, 12, 1) else 0.9)
            co = co_base * random.uniform(0.6, 1.5) * (0.7 if is_lockdown else 1.0)
            co = max(0.3, min(co, 8.5))

            # O3 (10-80 ug/m3): higher in sunny summer months, lower in winter/monsoon
            o3_base = 35.0 * (1.35 if d.month in (3, 4, 5) else (0.7 if d.month in (7, 8) else 0.95))
            o3 = o3_base * random.uniform(0.65, 1.45)
            o3 = max(8.0, min(o3, 85.0))

            # Calculate AQI based on dominant pollutant sub-index (PM2.5 & PM10 primarily)
            sub_pm25 = calculate_cpcb_subindex_pm25(pm25)
            sub_pm10 = calculate_cpcb_subindex_pm10(pm10)
            calculated_aqi = max(sub_pm25, sub_pm10) * random.uniform(0.97, 1.03)
            calculated_aqi = round(max(15.0, min(calculated_aqi, 500.0)), 1)
            calculated_bucket = get_aqi_bucket(calculated_aqi)

            # Convert values to strings or blanks for realistic missing data
            # Target: ~8-12% missing values distributed across pollutant columns
            val_pm25 = f"{pm25:.2f}" if random.random() > 0.08 else ""
            val_pm10 = f"{pm10:.2f}" if random.random() > 0.11 else ""
            val_no2 = f"{no2:.2f}" if random.random() > 0.09 else ""
            val_so2 = f"{so2:.2f}" if random.random() > 0.12 else ""
            val_co = f"{co:.2f}" if random.random() > 0.07 else ""
            val_o3 = f"{o3:.2f}" if random.random() > 0.10 else ""

            # Missing AQI scenario:
            # 1. If both PM2.5 and PM10 are missing, AQI cannot be reliably determined (~1% of rows)
            # 2. Or ~4-5% random sensor / reporting blackout where pollutants are present but AQI was not computed
            if (not val_pm25 and not val_pm10) or (random.random() < 0.045):
                val_aqi = ""
                val_bucket = ""
            else:
                val_aqi = f"{calculated_aqi:.1f}"
                val_bucket = calculated_bucket

            rows.append({
                "City": city,
                "Date": date_str,
                "PM2.5": val_pm25,
                "PM10": val_pm10,
                "NO2": val_no2,
                "SO2": val_so2,
                "CO": val_co,
                "O3": val_o3,
                "AQI": val_aqi,
                "AQI_Bucket": val_bucket
            })

    # Introduce ~0.1% duplicate rows (approx 25-30 duplicate rows across dataset)
    num_duplicates = max(10, int(len(rows) * 0.001))
    duplicate_indices = random.sample(range(len(rows)), num_duplicates)
    for idx in duplicate_indices:
        # Insert duplicate right after or randomly near
        rows.append(dict(rows[idx]))

    # Write out to CSV file
    fieldnames = ["City", "Date", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI", "AQI_Bucket"]
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully generated {len(rows)} records into {OUTPUT_FILE}")
    return len(rows)

if __name__ == "__main__":
    generate_data()
