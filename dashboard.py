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
st.title("🚲 Bike Sharing Analysis Dashboard")
st.markdown("""
This dashboard provides analysis of bike sharing data, exploring patterns and factors affecting bike rentals.
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

# Question 1: How do weather conditions affect bike rentals?
st.header("Question 1: How do weather conditions affect bike rentals?")
st.markdown("This analysis examines the relationship between weather conditions and bike rental patterns.")

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
    
    st.write(f"""
    **Observations:**
    - Clear weather has the highest average rentals ({weather_analysis.loc[weather_analysis['weathersit_label'] == 'Clear', 'mean'].values[0]:.0f} bikes).
    - Light Rain/Snow conditions have a significant negative impact on bike rentals, averaging only 1,803 rentals - approximately 63% lower than clear days.
    - There's a {(weather_analysis.loc[weather_analysis['weathersit_label'] == 'Clear', 'mean'].values[0] - weather_analysis.loc[weather_analysis['weathersit_label'] == 'Mist/Cloudy', 'mean'].values[0]) / weather_analysis.loc[weather_analysis['weathersit_label'] == 'Clear', 'mean'].values[0] * 100:.1f}% decrease in rentals from Clear to Mist/Cloudy conditions.
    """)

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
    
    best_season = seasonal_weather.loc[seasonal_weather['cnt'].idxmax()]
    st.write(f"""
    **Observations:**
    - Weather conditions affect rentals differently across seasons.
    - The highest average rentals occur during {best_season['season_label']} with {best_season['weathersit_label']} weather ({best_season['cnt']:.0f} bikes).
    - The negative impact of Light Rain/Snow is evident across all seasons, but varies in magnitude.
    - Even in bad weather, rentals remain relatively high during Summer compared to good weather in Winter.
    """)

# Question 2: What are the peak hours for bike rentals and how do they differ between weekdays and weekends?
st.header("Question 2: What are the peak hours for bike rentals and how do they differ between weekdays and weekends?")
st.markdown("This analysis examines hourly patterns and differences between weekdays and weekends.")

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
    
    # Find peak hours
    weekday_peak = hourly_pattern[hourly_pattern['is_weekend'] == 'Weekday'].sort_values('cnt', ascending=False).iloc[0]
    weekend_peak = hourly_pattern[hourly_pattern['is_weekend'] == 'Weekend'].sort_values('cnt', ascending=False).iloc[0]
    
    st.write(f"""
    **Observations:**
    - Weekdays show two distinct peak periods: morning (around {weekday_peak['hr']:.0f}:00) with {weekday_peak['cnt']:.0f} bikes and evening (17:00-18:00).
    - Weekends show a different pattern with a single peak around {weekend_peak['hr']:.0f}:00 with {weekend_peak['cnt']:.0f} bikes.
    -  Weekends maintain higher rental numbers throughout the midday period (11 AM to 5 PM).
    - Early morning (midnight to 6 AM) shows minimal activity on both weekdays and weekends.
    """)

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
    
    st.write("""
    **Observations:**
    - Registered users dominate overall rental volume, particularly on weekdays.
    - Casual users show much more consistent patterns between weekdays and weekends.
    - On weekends, registered users follow a single-peak pattern with maximum usage around midday (12-2 PM, ~240 rentals).
    - Casual users' rentals increase gradually throughout the day, peaking in mid-afternoon.
    - The weekday peaks for registered users are approximately 90% higher than their weekend peak.
    """)

# Additional Analysis
st.header("Additional Analysis: Seasonal and Monthly Trends")

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
    
    # Growth calculation
    if ('2011' in year_filter) and ('2012' in year_filter):
        yearly_avg = filtered_day_df.groupby('yr_label')['cnt'].mean()
        growth = (yearly_avg['2012'] - yearly_avg['2011']) / yearly_avg['2011'] * 100
        
        st.write(f"""
        **Observations:**
        - There's a clear seasonal pattern with higher rentals in warmer months.
        - 2012 shows consistently higher rental numbers compared to 2011, with approximately {growth:.1f}% growth.
        - The peak rental months are June-September, while the lowest are December-February.
        - This pattern suggests strong temperature and seasonality dependency in bike rentals.
        """)
    else:
        st.write("""
        **Observations:**
        - There's a clear seasonal pattern with higher rentals in warmer months.
        - The peak rental months are June-September, while the lowest are December-February.
        - This pattern suggests strong temperature and seasonality dependency in bike rentals.
        """)

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
    
    # Calculate seasonal variation
    casual_var = seasonal_user['casual'].max() / seasonal_user['casual'].min()
    registered_var = seasonal_user['registered'].max() / seasonal_user['registered'].min()
    
    st.write(f"""
    **Observations:**
    - Registered users significantly outnumber casual users across all seasons.
    - Casual users show {casual_var:.1f}x variation between peak and low seasons, while registered users show {registered_var:.1f}x.
    - Spring represents a significant low point for casual users.
    - Casual users display greater seasonal sensitivity than registered users.
    - Spring shows surprisingly low performance for both user types.
    """)

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
    
    st.write("""
    **Observations:**
    - Temperature is a major driver of bike rentals for all users.
    - Windspeed has a more significant deterrent effect than humidity.
    - Registered users show more consistent behavior patterns.
    - Registered Users : Extremely high correlation (0.95) with total rentals, indicating registered users drive the majority of business.
    - Casual Users: Strong correlation (0.67) with total rentals, but significantly less influential than registered users.
    """)

# Conclusions
st.header("Conclusions and Recommendations")
st.write("""
### Key Findings

#### The Impact of Weather on Bike Rentals

- **Clear Weather Conditions** yield the highest average rentals (4,877 bikes/day)
- **Light Rain/Snow** significantly decreases rentals (1,803 bikes/day, 63% lower than clear days)
- **Mist/Cloudy** shows reasonably good performance (4,036 bikes/day, only 17% lower than clear days)
- Data distribution is uneven: 463 clear days, 247 misty/cloudy days, and only 21 days with light rain/snow

#### Seasonal and Weather Variations

- **Fall Season** shows the highest rentals across all weather conditions
- **Summer** shows strong performance, particularly on clear days
- **Winter and Spring** generally have lower rental numbers
- Clear weather consistently outperforms other weather conditions in all seasons
- **Fall with Clear weather** combination shows the highest average rentals
- **Light Rain/Snow in spring** shows the lowest rental performance

#### Weekday vs Weekend Usage Patterns

- **Weekdays**: Two distinct peak periods at 8 AM and 5-6 PM (510 and 470 rentals)
- **Weekends**: A single broader peak spanning from around 11 AM to 5 PM (380 rentals)
- The maximum weekday peak is about 34% higher than the weekend peak
- Between 10 AM and 3 PM, weekend rentals substantially exceed weekday rentals
- Early morning (midnight to 6 AM) shows minimal activity on both day types

#### Behavior by User Type

- **Registered Users**:
  - Dominate overall rental volume, particularly on weekdays
  - Show two distinct commuting peaks on weekdays (morning 440 rentals, evening 460 rentals)
  - On weekends, follow a single-peak pattern (midday 240 rentals)
- **Casual Users**:
  - Show a single-peak pattern on both weekdays and weekends
  - Weekend usage significantly exceeds weekday usage
  - Peak rentals occur at 1-2 PM on weekends (140 rentals)
  - Display a leisure-oriented pattern regardless of the day type

#### Annual Growth

- Substantial increase in average rentals from 2011 to 2012 across all months
- Growth rate appears most significant in the earlier months (January-March)
- 2011 peaks in June (~4,800 rentals), while 2012 peaks in September (~7,250 rentals)
- The lowest rental periods occur in winter months (December-February)
- The average increase is approximately 2,000 additional rentals in 2012

#### Seasonal Patterns by User Type

- **Registered Users**:
  - Fall shows the highest average rentals (~4,400)
  - Spring has the lowest rentals (~2,300)
  - Winter and summer show similar performance (~4,000 rentals)
- **Casual Users**:
  - Fall and summer show the highest usage (~1,200 and ~1,100 respectively)
  - Spring has the lowest usage (~300)
  - Display greater seasonal sensitivity than registered users

#### Correlations with Weather Variables

- **Temperature**: Strong positive correlation (0.63) with total rentals
- **Windspeed**: Negative correlation (-0.23) with rentals, confirming riders are deterred by windy conditions
- **Humidity**: Slight negative correlation (-0.10) with rentals
- **Casual Users** are more strongly affected by temperature (0.54) than registered users
- **Registered Users** show stronger negative correlation with windspeed (-0.22)

#### Business Implications

1. **Differential Marketing Strategies**:
   - Commuter programs for registered users on weekdays
   - Recreational promotions for casual users, especially on weekends
   - Special strategies to boost rentals on light rain days

2. **Operational Planning**:
   - Bike availability should be optimized differently for weekdays versus weekends
   - Maintenance should be scheduled during low-demand periods (early morning hours)
   - Staffing can be adjusted based on weather forecasts

3. **Growth Opportunities**:
   - Spring represents the greatest opportunity for growth initiatives
   - The midday weekday period shows potential for growth through targeted promotions
   - Conversion strategies from casual to registered should focus on summer and fall

4. **Revenue Projections**:
   - Revenue projections can be significantly influenced by seasonal weather patterns
   - Registered users drive the majority of business (0.95 correlation with total rentals)
   - Weather forecasting could be integrated into business planning"
   """)