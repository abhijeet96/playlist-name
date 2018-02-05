import os
import pandas
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

X = []
labels = []
json_filepath = os.path.join('/','home','mm3','Desktop','RecSys','SpotifyAPI')

for filename in os.listdir(json_filepath):
		data = json.load(open(os.path.join(json_filepath, filename)))
		for track in data['audio_features']:
			track_features = []
			track_features.append(track['key'])
			track_features.append(track['energy'])
			track_features.append(track['valence'])
			track_features.append(track['speechiness'])
			track_features.append(track['loudness'])
			track_features.append(track['liveness'])
			track_features.append(track['danceability'])
			track_features.append(track['tempo'])
			track_features.append(track['acousticness'])
			track_features.append(track['instrumentalness'])
			labels.append(track['uri'])
			X.append(track_features)
range_n_clusters = [100,150,200,250,300,350,400,450,500]

X = np.asarray(X)
# Standarize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cluster_errors = []

for n_clusters in range_n_clusters:
	clusters = KMeans(n_clusters=n_clusters)
	clusters.fit(X_scaled)
	cluster_errors.append( clusters.inertia_ )
	# add other methods for error analysis such as silhouttes
	pred_classes = clusters.predict(X_scaled)
	
	cluster_dict = {}
	for cluster in range(n_clusters):
		track_uri_list = []
		for i in np.where(pred_classes == cluster):
			for j in i:
				track_uri_list.append(labels[j])
		cluster_dict[cluster] = track_uri_list

	with open( str(n_clusters) + 'clusters.json', 'w') as fp:
		json.dump(cluster_dict, fp, indent=4)

clusters_df = pd.DataFrame( { "num_clusters":range_n_clusters, "cluster_errors": cluster_errors } )
plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
plt.show()