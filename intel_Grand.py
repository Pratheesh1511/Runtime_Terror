import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import folium
import numpy as np

# Load the dataset from a CSV file
df = pd.read_csv('C:/Users/Pratheesh/Downloads/unnati_phase1_data_revised.csv')  # Replace 'your_dataset.csv' with the actual file path

# 1. Alert Analysis
alert_counts = df['Alert'].value_counts()
print("Alert Analysis:")
print(alert_counts)

# 2. Date and Time Analysis
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df['Hour'] = df['DateTime'].dt.hour
hourly_alert_counts = df.groupby('Hour')['Alert'].count()
print("\nHourly Alert Counts:")
print(hourly_alert_counts)

# 4. Vehicle Analysis
vehicle_counts = df['Vehicle'].nunique()
print("\nNumber of Unique Vehicles:")
print(vehicle_counts)

# 5. Speed Analysis
average_speed = df['Speed'].mean()
print("\nAverage Speed:")
print(average_speed)

# 6. Correlation Analysis (for numeric columns only)
numeric_columns = df.select_dtypes(include=['number'])
correlation_matrix = numeric_columns.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# 7. Time Series Analysis
daily_alert_counts = df.groupby('Date')['Alert'].count()
print("\nDaily Alert Counts:")
print(daily_alert_counts)

# Generate some sample data for the graphs
x = np.linspace(0, 10, 100)  # Example x-axis values
y1 = np.sin(x)  # Example y1 values (sine wave)
y2 = np.cos(x)  # Example y2 values (cosine wave)

frequently_involved_vehicles = df["Vehicle"].value_counts().sort_values(ascending=False).index[:3]
print("\nThe vehicles that are frequently involved in alerts are", frequently_involved_vehicles)

# Find locations where there are a high number of alerts
locations_with_high_number_of_alerts = df.groupby("Lat")["Alert"].count().sort_values(ascending=False).index[:3]
print("The locations where there are a high number of alerts are", locations_with_high_number_of_alerts)

# 9. Predictive Analysis (Example: Predicting Speed)
# Select relevant features and target variable (e.g., 'Speed')
X = df[['Lat', 'Long']]  # Replace with actual feature(s) you want to use
y = df['Speed']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nPredictive Analysis (Speed Prediction):")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
# 10. Summary Statistics
summary_stats = df.describe(include='all')
print("\nSummary Statistics:")
print(summary_stats)

# 11. Alert Mapping (requires geospatial data visualization libraries)
# Map alerts to specific locations and visualize them on a map.

# Load the dataset from a CSV file  # Replace 'your_dataset.csv' with the actual file path

# Create a map centered on a location (you can customize the coordinates)
latitude_center = df['Lat'].mean()
longitude_center = df['Long'].mean()

m = folium.Map(location=[latitude_center, longitude_center], zoom_start=10)

# Iterate through the dataset and add markers for each alert
for index, row in df.iterrows():
    lat, lon = row['Lat'], row['Long']
    alert_type = row['Alert']
    folium.Marker([lat, lon], tooltip=alert_type).add_to(m)

# Save the map to an HTML file
m.save('alert_map.html')