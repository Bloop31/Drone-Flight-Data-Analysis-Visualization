import numpy as np
import os   
import pandas as pd
import matplotlib.pyplot as plt

file_path=r"C:\Users\Prajjit Basu\OneDrive\Desktop\python_proj\droneds.csv"
df=pd.read_csv(file_path, on_bad_lines="skip")

df['Flight Date'] = pd.to_datetime(df['Flight Date'])
df['Battery Remaining (%)'] = pd.to_numeric(df['Battery Remaining (%)'], errors='coerce')

df=df.dropna(subset=['Altitude (meters)', 'Flight Duration (minutes)', 'Distance Flown (km)', 'Battery Remaining (%)'])

print("\nBasic statistics:")
print(df[['Altitude (meters)', 'Flight Duration (minutes)', 
          'Distance Flown (km)', 'Battery Remaining (%)']].describe())

plt.figure(figsize=(8,5))
plt.scatter(df['Distance Flown (km)'], df['Battery Remaining (%)'], 
            c=df['Altitude (meters)'], cmap='viridis', alpha=0.7)
plt.colorbar(label="Altitude (m)")
plt.xlabel("Distance Flown (km)")
plt.ylabel("Battery Remaining (%)")
plt.title("Battery Remaining vs Distance (colored by Altitude)")
plt.show()

df['Battery Used (%)'] = 100 - df['Battery Remaining (%)']
df['Drain per Minute'] = df['Battery Used (%)'] / df['Flight Duration (minutes)']
df['Drain per Km'] = df['Battery Used (%)'] / df['Distance Flown (km)']

print("\nAverage drain per minute: ", df['Drain per Minute'].mean())
print("Average drain per km: ", df['Drain per Km'].mean())

plt.figure(figsize=(8,5))
plt.hist(df['Drain per Minute'], bins=20, color='orange', alpha=0.7)
plt.xlabel("Battery Drain per Minute (%)")
plt.ylabel("Number of Flights")
plt.title("Distribution of Battery Efficiency")
plt.show()

corr = df[['Altitude (meters)', 'Flight Duration (minutes)', 
           'Distance Flown (km)', 'Battery Used (%)']].corr()

print("\nCorrelation matrix:")
print(corr)

# Heatmap-like visualization
plt.figure(figsize=(6,5))
plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar(label="Correlation")
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# 6. Anomaly Detection
# -------------------------------
mean_drain = df['Drain per Minute'].mean()
std_drain = df['Drain per Minute'].std()

# Define anomaly: drain more than mean + 2*std
df['Anomaly'] = df['Drain per Minute'] > (mean_drain + 2*std_drain)
anomalies = df[df['Anomaly']]

print("\nFlights with abnormal battery drain:")
print(anomalies[['Drone ID', 'Altitude (meters)', 'Flight Duration (minutes)', 
                 'Distance Flown (km)', 'Drain per Minute']])

# Plot anomalies
plt.figure(figsize=(8,5))
plt.scatter(df['Flight Duration (minutes)'], df['Drain per Minute'], label="Normal")
plt.scatter(anomalies['Flight Duration (minutes)'], anomalies['Drain per Minute'], 
            color='red', label="Anomaly")
plt.xlabel("Flight Duration (minutes)")
plt.ylabel("Battery Drain per Minute (%)")
plt.title("Anomaly Detection in Battery Efficiency")
plt.legend()
plt.show()