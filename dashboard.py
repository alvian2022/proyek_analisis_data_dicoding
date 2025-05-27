import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Bike Sharing Dashboard",
    page_icon="🚲",
    layout="wide"
)

# Set matplotlib style to match original formatting

# Function to load data
@st.cache_data
def load_data():
    # Load day and hour data
    day_df = pd.read_csv('data/day.csv')
    hour_df = pd.read_csv('data/hour.csv')
    
    # Convert date strings to datetime
    day_df['dteday'] = pd.to_datetime(day_df['dteday'])
    hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])
    
    # Create datetime column for hour data
    hour_df['datetime'] = pd.to_datetime(hour_df['dteday']) + pd.to_timedelta(hour_df['hr'], unit='h')
    
    # Map categorical variables to their meaningful representations
    season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    weather_map = {1: 'Clear', 2: 'Mist/Cloudy', 3: 'Light Rain/Snow', 4: 'Heavy Rain/Snow'}
    year_map = {0: '2011', 1: '2012'}
    
    day_df['season_label'] = day_df['season'].map(season_map)
    day_df['weathersit_label'] = day_df['weathersit'].map(weather_map)
    day_df['yr_label'] = day_df['yr'].map(year_map)
    
    hour_df['season_label'] = hour_df['season'].map(season_map)
    hour_df['weathersit_label'] = hour_df['weathersit'].map(weather_map)
    hour_df['yr_label'] = hour_df['yr'].map(year_map)
    hour_df['is_weekend'] = hour_df['weekday'].apply(lambda x: 'Weekend' if x in [0, 6] else 'Weekday')
    
    return day_df, hour_df

# Load data
day_df, hour_df = load_data()

# Title and description
st.title("🚲 Bike Sharing Visualization Dashboard")
st.markdown("""
This dashboard provides visualizations of bike sharing data, exploring patterns and factors affecting bike rentals.
""")

# Show dataset info
with st.expander("About the Dataset"):
    st.write("""
    The Bike Sharing dataset contains daily and hourly counts of rental bikes in a bike-sharing system with corresponding weather and seasonal information.
    
    **Features:**
    - **datetime**: Date and hour (for hour dataset)
    - **season**: Season (1:springer, 2:summer, 3:fall, 4:winter)
    - **holiday**: Whether it's a holiday or not
    - **workingday**: Whether it's a working day or not
    - **weathersit**: Weather situation (1:Clear, 2:Mist/Cloudy, 3:Light Rain/Snow, 4:Heavy Rain/Snow)
    - **temp/atemp**: Temperature/feeling temperature
    - **hum**: Humidity
    - **windspeed**: Wind speed
    - **casual**: Count of casual users
    - **registered**: Count of registered users
    - **cnt**: Total count of rentals
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Daily Data Sample")
        st.dataframe(day_df.head(3))
    with col2:
        st.subheader("Hourly Data Sample")
        st.dataframe(hour_df.head(3))

# Sidebar for filters
st.sidebar.header("Filters")

# Year filter
year_filter = st.sidebar.multiselect(
    "Select Year",
    options=day_df['yr_label'].unique(),
    default=day_df['yr_label'].unique()
)

# Season filter
season_filter = st.sidebar.multiselect(
    "Select Season",
    options=day_df['season_label'].unique(),
    default=day_df['season_label'].unique()
)

# Weather filter
weather_filter = st.sidebar.multiselect(
    "Select Weather",
    options=day_df['weathersit_label'].unique(),
    default=day_df['weathersit_label'].unique()
)

# Apply filters
filtered_day_df = day_df[
    (day_df['yr_label'].isin(year_filter)) &
    (day_df['season_label'].isin(season_filter)) &
    (day_df['weathersit_label'].isin(weather_filter))
]

filtered_hour_df = hour_df[
    (hour_df['yr_label'].isin(year_filter)) &
    (hour_df['season_label'].isin(season_filter)) &
    (hour_df['weathersit_label'].isin(weather_filter))
]

# Key metrics cards
st.header("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_rentals = filtered_day_df['cnt'].sum()
    st.metric("Total Bike Rentals", f"{total_rentals:,}")

with col2:
    avg_daily_rentals = filtered_day_df['cnt'].mean()
    st.metric("Average Daily Rentals", f"{avg_daily_rentals:.1f}")

with col3:
    casual_percentage = (filtered_day_df['casual'].sum() / total_rentals) * 100
    st.metric("Casual Riders", f"{casual_percentage:.1f}%")

with col4:
    registered_percentage = (filtered_day_df['registered'].sum() / total_rentals) * 100
    st.metric("Registered Riders", f"{registered_percentage:.1f}%")

# Business Question 1: Weather Impact Analysis
st.header("1. How do weather conditions impact bike rentals across different seasons?")

plt.figure(figsize=(12, 7))
sns.boxplot(x='season_label', y='cnt', hue='weathersit_label', data=filtered_day_df)
plt.title('Bike Rentals by Season and Weather Condition')
plt.xlabel('Season')
plt.ylabel('Number of Rentals')
plt.xticks(rotation=0)
plt.legend(title='Weather', loc='upper left')
plt.tight_layout()
st.pyplot(plt)

# Business Question 2: Peak Hours Analysis
st.header("2. What are the peak hours for bike rentals and how do they differ between weekdays and weekends?")

hourly_pattern = filtered_hour_df.groupby(['hr', 'is_weekend'])['cnt'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(x='hr', y='cnt', hue='is_weekend', data=hourly_pattern, marker='o')
plt.title('Average Hourly Bike Rentals: Weekdays vs. Weekends')
plt.xlabel('Hour of Day')
plt.ylabel('Average Number of Rentals')
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Day Type')
plt.tight_layout()
st.pyplot(plt)

# Business Question 3: User Type Analysis
st.header("3. How do usage patterns differ between casual and registered users throughout the day?")

hourly_by_user = filtered_hour_df.groupby(['hr', 'is_weekend'])[['casual', 'registered']].mean().reset_index()
hourly_by_user_melted = pd.melt(hourly_by_user,
                                id_vars=['hr', 'is_weekend'],
                                value_vars=['casual', 'registered'],
                                var_name='user_type',
                                value_name='avg_count')

plt.figure(figsize=(14, 7))
sns.lineplot(x='hr', y='avg_count', hue='user_type', style='is_weekend', data=hourly_by_user_melted, marker='o')
plt.title('Average Hourly Bike Rentals by User Type: Weekdays vs. Weekends')
plt.xlabel('Hour of Day')
plt.ylabel('Average Number of Rentals')
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='User Type / Day')
plt.tight_layout()
st.pyplot(plt)

# Business Question 4: Monthly and Yearly Trends
st.header("4. How do bike rentals fluctuate throughout the year and between different years?")

monthly_trends = filtered_day_df.groupby(['mnth', 'yr_label'])['cnt'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(x='mnth', y='cnt', hue='yr_label', data=monthly_trends, marker='o')
plt.title('Average Monthly Bike Rentals by Year')
plt.xlabel('Month')
plt.ylabel('Average Number of Rentals')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Year')
plt.tight_layout()
st.pyplot(plt)

# Seasonal User Analysis
st.header("Average Seasonal Bike Rentals by User Type")

seasonal_user_trends = filtered_day_df.groupby('season_label')[['casual', 'registered', 'cnt']].mean().reset_index()
seasonal_user_melted = pd.melt(seasonal_user_trends,
                              id_vars=['season_label'],
                              value_vars=['casual', 'registered'],
                              var_name='user_type',
                              value_name='avg_count')

plt.figure(figsize=(10, 6))
sns.barplot(x='season_label', y='avg_count', hue='user_type', data=seasonal_user_melted)
plt.title('Average Seasonal Bike Rentals by User Type')
plt.xlabel('Season')
plt.ylabel('Average Number of Rentals')
plt.xticks(rotation=0)
plt.legend(title='User Type')
plt.tight_layout()
st.pyplot(plt)

# Correlation Analysis
st.header("Correlation Analysis")

correlation = filtered_day_df[['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']].corr()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
st.pyplot(plt)