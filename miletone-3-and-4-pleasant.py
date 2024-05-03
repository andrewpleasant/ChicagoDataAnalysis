# Import necessary libraries
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

# Define API endpoints and tokens
crash_endpoint = "https://data.cityofchicago.org/resource/85ca-t3if.json"
taxi_endpoint = "https://data.cityofchicago.org/resource/wrvz-psew.json"
api_token = "QF1cCNCQ4u1I87WMuIpVHOkH7"

# Define limit for retrieved data per page
limit = 300000

# Define an empty list to store retrieved data points
all_data = []

# Function to retrieve data from API
def fetch_data(endpoint, select_columns=None):
    all_data = []
    url = f"{endpoint}?$limit={limit}&$offset=0"
    if select_columns:
        url += f"&$select={','.join(select_columns)}"
    headers = {"X-App-Token": api_token}

    try:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            all_data.extend(data)
            print(f"Retrieved {len(data)} records from {endpoint}")
            return pd.DataFrame(all_data)
        else:
            print(f"Error retrieving data from {endpoint}: {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        print(f"An error occurred while retrieving data from {endpoint}: {e}")
        return pd.DataFrame()

# Fetch crash data
crash_master_df = fetch_data(crash_endpoint, select_columns=['crash_record_id', 'crash_date'])

# Fetch taxi trip data
taxi_master_df = fetch_data(taxi_endpoint)

# Clean crash data
crash_working_df = crash_master_df.copy()
print(f"Cleaned crash data: {crash_working_df.shape[0]} rows remaining")

# Clean taxi trip data
taxi_working_df = taxi_master_df.dropna(subset=['trip_end_timestamp'])
print(f"Cleaned taxi trip data: {taxi_working_df.shape[0]} rows remaining")

# Filter crash data for 2020-2024
start_date = "2020-01-01"
end_date = "2024-12-31"

# Filter taxi trip data for 2020-2024
taxi_working_df = taxi_working_df[(taxi_working_df['trip_end_timestamp'] >= start_date) & (taxi_working_df['trip_end_timestamp'] <= end_date)]
taxi_working_df['year'] = pd.to_datetime(taxi_working_df['trip_end_timestamp']).dt.year

# Calculate yearly crashes
yearly_crashes = crash_working_df.groupby(pd.to_datetime(crash_working_df['crash_date']).dt.year)['crash_record_id'].nunique().reset_index()
yearly_crashes.columns = ['year', 'crashes']

# Calculate yearly taxi trips
yearly_trips = taxi_working_df.groupby('year').size().reset_index()
yearly_trips.columns = ['year', 'trips']


# Number of crashes per year (2020-2024)
plt.figure(figsize=(10, 6))
plt.bar(yearly_crashes['year'], yearly_crashes['crashes'])
plt.xlabel("Year")
plt.ylabel("Number of Crashes")
plt.title("Number of Crashes per Year (2020-2024)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Number of taxi trips per year (2020-2024)
plt.figure(figsize=(10, 6))
plt.bar(yearly_trips['year'], yearly_trips['trips'])
plt.xlabel("Year")
plt.ylabel("Number of Trips")
plt.title("Number of Taxi Trips per Year (2020-2024)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Combined plot showing crashes and taxi trips
plt.figure(figsize=(10, 6))
plt.bar(yearly_crashes['year'], yearly_crashes['crashes'], label='Crashes')
plt.bar(yearly_trips['year'], yearly_trips['trips'], alpha=0.5, label='Taxi Trips')
plt.xlabel("Year")
plt.ylabel("Number of Crashes/Trips")
plt.title("Crashes and Taxi Trips per Year (2020-2024)")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.show()

# Statistical Analysis
print("\nStatistical Analysis:\n")

# statistics for crashes
print("Descriptive Statistics for Crashes:")
print(yearly_crashes['crashes'].describe())

# statistics for taxi trips
print("\nDescriptive Statistics for Taxi Trips:")
print(yearly_trips['trips'].describe())

# Mann-Whitney U Test - Source:https://leansigmacorporation.com/mann-whitney-testing-with-minitab/#:~:text=The%20Mann%E2%80%93Whitney%20test%20can,two%20populations'%20distributions%20are%20different.
print("\nMann-Whitney U Test:")
statistic, p_val = mannwhitneyu(yearly_crashes['crashes'], yearly_trips['trips'])
print(f"Test statistic: {statistic:.2f}")
print(f"p-value: {p_val:.4f}")

# Check if the distributions are significantly different at alpha = 0.05
alpha = 0.05
if p_val < alpha:
    print("The distributions of crashes and taxi trips are significantly different.")
else:
    print("The distributions of crashes and taxi trips are not significantly different.")

# graph the distributions
plt.figure(figsize=(10, 6))
plt.boxplot([yearly_crashes['crashes'], yearly_trips['trips']], labels=['Crashes', 'Taxi Trips'])
plt.ylabel("Number of Crashes/Trips")
plt.title("Distribution of Crashes and Taxi Trips")
plt.tight_layout()
plt.show()
