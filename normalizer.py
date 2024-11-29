import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns

#load the data
df = pd.read_csv(r'data\adult-data-new.csv')

#explore the data
print(df.info())
print(df.describe())

#preprocess
#categorical variables
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                    'relationship', 'race', 'sex']
encoder = OneHotEncoder(sparse=False)
encoded_data = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())

#combine encoded data above with original numerical columns from df data
df_encoded = pd.concat([df[['age', 'capital-gain', 'capital-loss', 'hours-per-week']], encoded_df], axis=1)

#normalize numerical features
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df_encoded)
normalized_df = pd.DataFrame(normalized_data, columns=df_encoded.columns)

#determine the optimal number of clusters
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_data)
    silhouette_avg = silhouette_score(normalized_data, cluster_labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores)
plt.title("Silhouette Score vs Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()

#apply k-means clustering with optimal number of clusters
optimal_n_clusters = 5  # Based on the elbow/silhouette score plot
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(normalized_data)

#add cluster labels to original dataframe
df['Cluster'] = cluster_labels

#visualize results using plotgraphs and scatterplots
plt.figure(figsize=(10, 8))
sns.scatterplot(x='capital-gain', y='hours-per-week', hue='Cluster', data=df, palette='viridis')
plt.title('K-means Clustering Results')
plt.xlabel('Capital Gain')
plt.ylabel('Hours per Week')
plt.legend(title='Cluster')
plt.show()

#analyze clusters based on salary
for cluster in range(optimal_n_clusters):
    cluster_df = df[df['Cluster'] == cluster]
    print(f"\nCluster {cluster}:")
    print(cluster_df[['Salary', 'capital-gain', 'hours-per-week']].describe())
