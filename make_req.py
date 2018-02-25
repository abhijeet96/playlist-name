import requests
import json
import pandas as pd
import numpy as np
import os
import sys
import csv

json_filepath = os.path.join('/','media','avi','External HDD','Data - RecSys','data')
#json_filepath = os.path.join('/','home','avi','RecSys','playlist-name','Data')
#Need to make a list of unique tracks. dictionary wont do. and then run this

uri_list=[]

track_uri_to_id={}

i=0

with open('uri_list', 'w+') as myfile:
	wr = csv.writer(myfile)

	#print(json_filepath)
	for file in os.listdir(json_filepath):

		#print(file)

		data = json.load(open(os.path.join(json_filepath,file)))

		playlists= data['playlists']

		pass_this_to_api=''

		j=0		

		for playlist in playlists:
			name = playlist['name']
			tracks= playlist['tracks']
			pid=playlist['pid']		

			for track in tracks:
				track_uri=track['track_uri']
				#can add name and album later if needed
				if not track_uri in track_uri_to_id:
					track_uri_to_id[track_uri]=i
					#uri_list.append(track_uri.split(':')[2])
					wr.writerow([track_uri.split(':')[2]])
					i+=1
					print(i)			
    				
