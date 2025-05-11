import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Mall_Customers.csv")

df_encoded = df.copy()
df_encoded['Gender'] = df_encoded['Gender'].map({'Male': 0, 'Female': 1})
X = df_encoded.drop('CustomerID', axis=1)

kmeans1 = KMeans(n_clusters=5, random_state=42)
labels1 = kmeans1.fit_predict(X)
df_encoded['Cluster_no_scale'] = labels1

X_scaled = X.copy()
scaler = StandardScaler()
X_scaled[['Annual Income (k$)', 'Spending Score (1-100)']] = scaler.fit_transform(X_scaled[['Annual Income (k$)', 'Spending Score (1-100)']])
kmeans2 = KMeans(n_clusters=5, random_state=42)
labels2 = kmeans2.fit_predict(X_scaled)
df_encoded['Cluster_scaled'] = labels2

print(df_encoded[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster_no_scale', 'Cluster_scaled']].head())
