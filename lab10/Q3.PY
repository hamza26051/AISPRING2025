import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

student_data = {
    'student_id': [101, 102, 103, 104, 105, 106, 107, 108],
    'GPA': [3.8, 2.5, 3.2, 2.0, 3.6, 2.7, 3.9, 2.8],
    'study_hours': [15, 5, 10, 3, 14, 6, 16, 7],
    'attendance_rate': [95, 60, 80, 50, 90, 65, 97, 70]
}
df_students = pd.DataFrame(student_data)

X_student = df_students[['GPA', 'study_hours', 'attendance_rate']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_student)

wcss = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.plot(range(2, 7), wcss, marker='o')
plt.title("Elbow Method - Optimal K")
plt.xlabel("No. of Clusters (K)")
plt.ylabel("WCSS")
plt.show()

kmeans_final = KMeans(n_clusters=3, random_state=42)
df_students['Cluster'] = kmeans_final.fit_predict(X_scaled)

plt.figure(figsize=(8,5))
for i in range(3):
    cluster = df_students[df_students['Cluster'] == i]
    plt.scatter(cluster['study_hours'], cluster['GPA'], label=f"Cluster {i}")
plt.xlabel("Study Hours")
plt.ylabel("GPA")
plt.title("Student Clusters by Study Hours and GPA")
plt.legend()
plt.grid(True)
plt.show()

print(df_students[['student_id', 'GPA', 'study_hours', 'attendance_rate', 'Cluster']])
