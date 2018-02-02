import pandas as pd
import numpy as np
import os
import json
import sys

json_filepath = os.path.join('mpd.slice.0-999.json')

data = json.load(open(json_filepath))

playlists= data['playlists']

track_uri_to_id={}
id_to_track_uri={}

i=0
j=0
pid_to_playlist_name={}
playlist_matrix={}

for playlist in playlists:
	name = playlist['name']
	tracks= playlist['tracks']
	pid=playlist['pid']

	pid_to_playlist_name[pid]=name

	track_uri_list=[]

	for track in tracks:
		track_uri=track['track_uri']
		#can add name and album later if needed
		if not track_uri in track_uri_to_id:
			track_uri_to_id[track_uri]=i
			id_to_track_uri[i]=track_uri
			i+=1		

		track_uri_list.append(track_uri_to_id[track_uri])

	playlist_matrix[pid]=track_uri_list

	#print("playlist no.", j)
	j+=1
print(playlist_matrix)
