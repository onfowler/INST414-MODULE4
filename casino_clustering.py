import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("Loading dataset and sampling 20,000 rows to prevent memory crashes...")
df = pd.read_csv('online_casino_games_dataset_v2.csv')
df = df.sample(n=20000, random_state=42)

# Standardize column names to lowercase 
df.columns = df.columns.str.lower()

# 1. Feature Selection & Cleaning
# UPDATED: Using 'max_win' instead of 'max_multiplier' based on your actual columns
features = ['rtp', 'min_bet', 'max_win']
print(f"Extracting features for similarity measurement: {features}")

# Drop rows where these specific features might be entirely blank
clean_df = df.dropna(subset=features).copy()

# 2. Scaling the Data
# We MUST scale the data because max_win can be in the hundreds of thousands, while RTP is just ~96
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clean_df[features])

# 3. Selecting K (The Elbow Method)
print("Calculating the Elbow Method to find optimal K...")
inertia = []
k_values = range(1, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Save the Elbow Plot for the Medium post
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Casino Game Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Distance between points)')
plt.grid(True)
plt.savefig('elbow_plot.png')
print("Saved 'elbow_plot.png' to your folder!")

# 4. Applying K-Means Clustering
optimal_k = 3
print(f"\nApplying K-Means with k={optimal_k}...")
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clean_df['cluster'] = final_kmeans.fit_predict(scaled_data)

# 5. Analyzing the Clusters
print("\n--- Cluster Profiles ---")
for i in range(optimal_k):
    cluster_data = clean_df[clean_df['cluster'] == i]
    print(f"\nCluster {i} (Contains {len(cluster_data)} games)")
    print(f"Average RTP: {cluster_data['rtp'].mean():.2f}%")
    print(f"Average Min Bet: ${cluster_data['min_bet'].mean():.2f}")
    
    # UPDATED: Printing Max Win
    print(f"Average Max Win: ${cluster_data['max_win'].mean():,.2f}")
    
    # Print two examples from this cluster using your actual column name 'game'
    print("Examples:")
    examples = cluster_data.head(2)
    for index, row in examples.iterrows():
        print(f" - {row['game']} ({row['game_type']})")