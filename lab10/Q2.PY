import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

vehicle_data = {
    'vehicle_serial_no': [5, 3, 8, 2, 4, 7, 6, 10, 1, 9],
    'mileage': [150000, 120000, 250000, 80000, 100000, 220000, 180000, 300000, 75000, 280000],
    'fuel_efficiency': [15, 18, 10, 22, 20, 12, 16, 8, 24, 9],
    'maintenance_cost': [5000, 4000, 7000, 2000, 3000, 6500, 5500, 8000, 1500, 7500],
    'vehicle_type': ['SUV', 'Sedan', 'Truck', 'Hatchback', 'Sedan', 'Truck', 'SUV', 'Truck', 'Hatchback', 'SUV']
}
df_vehicle = pd.DataFrame(vehicle_data)

df_vehicle['vehicle_type_encoded'] = df_vehicle['vehicle_type'].astype('category').cat.codes
X_vehicle = df_vehicle[['mileage', 'fuel_efficiency', 'maintenance_cost', 'vehicle_type_encoded']]

kmeans_ns = KMeans(n_clusters=3, random_state=42)
df_vehicle['Cluster_no_scale'] = kmeans_ns.fit_predict(X_vehicle)

scaler = StandardScaler()
X_scaled = X_vehicle.copy()
X_scaled[['mileage', 'fuel_efficiency', 'maintenance_cost']] = scaler.fit_transform(X_scaled[['mileage', 'fuel_efficiency', 'maintenance_cost']])
kmeans_s = KMeans(n_clusters=3, random_state=42)
df_vehicle['Cluster_scaled'] = kmeans_s.fit_predict(X_scaled)

print(df_vehicle[['vehicle_serial_no', 'mileage', 'fuel_efficiency', 'maintenance_cost', 'vehicle_type', 'Cluster_no_scale', 'Cluster_scaled']])
