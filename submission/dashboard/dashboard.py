import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Bike Sharing Dashboard",
    page_icon="🚲",
    layout="wide"
)

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

# Weather Impact Visualizations
st.header("Weather Impact on Bike Rentals")

# Create tabs for different visualizations
tab1, tab2 = st.tabs(["Weather Impact", "Weather by Season"])

with tab1:
    # Average rentals by weather condition
    weather_analysis = filtered_day_df.groupby('weathersit_label')['cnt'].agg(['mean', 'count']).reset_index()
    
    fig = px.bar(
        weather_analysis, 
        x='weathersit_label', 
        y='mean',
        color='weathersit_label',
        labels={'weathersit_label': 'Weather Condition', 'mean': 'Average Rentals'},
        title="Average Bike Rentals by Weather Condition",
        text_auto='.0f'
    )
    fig.update_layout(xaxis_title="Weather Condition", yaxis_title="Average Number of Rentals")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Effect of weather across seasons
    seasonal_weather = filtered_day_df.groupby(['season_label', 'weathersit_label'])['cnt'].mean().reset_index()
    
    fig = px.bar(
        seasonal_weather,
        x='season_label',
        y='cnt',
        color='weathersit_label',
        barmode='group',
        labels={'season_label': 'Season', 'cnt': 'Average Rentals', 'weathersit_label': 'Weather'},
        title="Average Bike Rentals by Season and Weather",
        category_orders={"season_label": ["Winter", "Spring", "Summer", "Fall"]}
    )
    fig.update_layout(xaxis_title="Season", yaxis_title="Average Number of Rentals")
    st.plotly_chart(fig, use_container_width=True)

# Hourly Patterns Visualizations
st.header("Hourly Rental Patterns")

# Create tabs for different visualizations
tab1, tab2 = st.tabs(["Hourly Patterns", "User Type Analysis"])

with tab1:
    # Hourly rental patterns - weekday vs weekend
    hourly_pattern = filtered_hour_df.groupby(['hr', 'is_weekend'])['cnt'].mean().reset_index()
    
    fig = px.line(
        hourly_pattern, 
        x='hr', 
        y='cnt', 
        color='is_weekend',
        labels={'hr': 'Hour of Day', 'cnt': 'Average Rentals', 'is_weekend': 'Day Type'},
        title="Average Hourly Bike Rentals: Weekdays vs. Weekends",
        markers=True
    )
    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=1),
        xaxis_title="Hour of Day (24-hour format)",
        yaxis_title="Average Number of Rentals",
        legend_title="Day Type"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Hourly pattern by user type (casual vs. registered)
    hourly_user = filtered_hour_df.groupby(['hr', 'is_weekend'])[['casual', 'registered']].mean().reset_index()
    hourly_user_melted = pd.melt(
        hourly_user, 
        id_vars=['hr', 'is_weekend'], 
        value_vars=['casual', 'registered'],
        var_name='user_type', 
        value_name='avg_count'
    )
    
    fig = px.line(
        hourly_user_melted, 
        x='hr', 
        y='avg_count', 
        color='user_type',
        line_dash='is_weekend',
        labels={'hr': 'Hour of Day', 'avg_count': 'Average Rentals', 'user_type': 'User Type', 'is_weekend': 'Day Type'},
        title="Average Hourly Bike Rentals by User Type: Weekdays vs. Weekends",
        markers=True
    )
    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=1),
        xaxis_title="Hour of Day (24-hour format)",
        yaxis_title="Average Number of Rentals",
        legend_title="User Type / Day"
    )
    st.plotly_chart(fig, use_container_width=True)

# Seasonal and Monthly Trends
st.header("Seasonal and Monthly Trends")

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Monthly Trends", "Seasonal User Trends", "Correlation Analysis"])

with tab1:
    # Monthly trends analysis
    monthly_trends = filtered_day_df.groupby(['mnth', 'yr_label'])['cnt'].mean().reset_index()
    
    fig = px.line(
        monthly_trends, 
        x='mnth', 
        y='cnt', 
        color='yr_label',
        labels={'mnth': 'Month', 'cnt': 'Average Rentals', 'yr_label': 'Year'},
        title="Average Monthly Bike Rentals by Year",
        markers=True
    )
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ),
        xaxis_title="Month",
        yaxis_title="Average Number of Rentals",
        legend_title="Year"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Seasonal user trends
    seasonal_user = filtered_day_df.groupby('season_label')[['casual', 'registered', 'cnt']].mean().reset_index()
    seasonal_user_melted = pd.melt(
        seasonal_user, 
        id_vars=['season_label'], 
        value_vars=['casual', 'registered'],
        var_name='user_type', 
        value_name='avg_count'
    )
    
    fig = px.bar(
        seasonal_user_melted, 
        x='season_label', 
        y='avg_count', 
        color='user_type',
        barmode='group',
        labels={'season_label': 'Season', 'avg_count': 'Average Rentals', 'user_type': 'User Type'},
        title="Average Seasonal Bike Rentals by User Type",
        category_orders={"season_label": ["Winter", "Spring", "Summer", "Fall"]}
    )
    fig.update_layout(
        xaxis_title="Season",
        yaxis_title="Average Number of Rentals",
        legend_title="User Type"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Correlation analysis
    corr_vars = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
    correlation = filtered_day_df[corr_vars].corr()
    
    fig = px.imshow(
        correlation,
        labels=dict(x="Feature", y="Feature", color="Correlation"),
        x=corr_vars,
        y=corr_vars,
        title="Correlation Matrix of Numerical Features",
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )
    fig.update_layout(
        width=700,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show correlation with rentals
    corr_with_cnt = correlation['cnt'].sort_values(ascending=False).drop('cnt')
    
    fig = px.bar(
        x=corr_with_cnt.index,
        y=corr_with_cnt.values,
        labels={'x': 'Feature', 'y': 'Correlation with Rentals'},
        title="Features Correlation with Total Rentals",
        color=corr_with_cnt.values,
        color_continuous_scale='RdBu_r',
        text_auto='.2f'
    )
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Correlation Coefficient",
        yaxis=dict(range=[-1, 1])
    )
    st.plotly_chart(fig, use_container_width=True)