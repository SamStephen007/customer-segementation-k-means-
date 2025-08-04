import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import silhouette_score


df = pd.read_csv("Mall_Customers.csv")
df.head()

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
# After kmeans = KMeans(...), and df['Cluster'] = kmeans.fit_predict(...)

# Cluster-wise average income and spending
cluster_means = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("\nCluster Centers:\n", cluster_means)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    data=df,
    palette='Set2',
    s=100
)
plt.title("Customer Segmentation with KMeans")
plt.show()

print(df['Cluster'].value_counts())

silhouette = silhouette_score(X_scaled, df['Cluster'])
print("Silhouette Score:", round(silhouette, 3))

import pickle

import pickle
pickle.dump(kmeans, open("kmeans_model.pickle", "wb"))
pickle.dump(scaler, open("scaler.pickle", "wb"))


print("Model saved successfully as kmeans_model.pickle")

features = ['Annual Income (k$)', 'Spending Score (1-100)']


new_data_df = pd.DataFrame([[60, 70]], columns=features)

new_data_scaled = scaler.transform(new_data_df)
predicted_cluster = kmeans.predict(new_data_scaled)[0]

print("Predicted Cluster:", predicted_cluster)

