import os
import pandas
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

json_filepath = os.path.join('/','home','mm3','Desktop','RecSys','SpotifyAPI')

track_features_key=[]
track_features_energy=[]
track_features_valence=[]
track_features_speechiness=[]
track_features_loudness=[]
track_features_liveness=[]
track_features_danceability=[]
track_features_speechiness=[]
track_features_tempo=[]
track_features_acousticness=[]
track_features_instrumentalness=[]


for filename in os.listdir(json_filepath):
	data = json.load(open(os.path.join(json_filepath, filename)))
	#print(filename)
# data = json.load(open(os.path.join(json_filepath, "100-199.json")))

	for track in data['audio_features']:
		try:
			track_features_key.append(track['key'])
			track_features_energy.append(track['energy'])
			track_features_valence.append(track['valence'])
			track_features_speechiness.append(track['speechiness'])
			track_features_loudness.append(track['loudness'])
			track_features_liveness.append(track['liveness'])
			track_features_danceability.append(track['danceability'])
			track_features_tempo.append(track['tempo'])
			track_features_acousticness.append(track['acousticness'])
			track_features_instrumentalness.append(track['instrumentalness'])
		except Exception as e:
			print(filename)

n, bins, patches = plt.hist(track_features_energy, facecolor='g', alpha=0.75)
hist=np.histogram(track_features_energy)
print (n,bins)
plt.title('Energy')
plt.show()
n, bins, patches = plt.hist(track_features_valence, facecolor='r', alpha=0.1)
plt.title('valence')
plt.show()