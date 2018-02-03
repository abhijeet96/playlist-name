import requests
import json
import pandas as pd
import numpy as np
import os
import sys

json_filepath = os.path.join('mpd.slice.0-999.json')

#Need to make a list of unique tracks. dictionary wont do. and then run this

data = json.load(open(json_filepath))

playlists= data['playlists']

track_uri_to_id={}
pass_this_to_api=''

i=0
j=0
pid_to_playlist_name={}

for playlist in playlists:
	name = playlist['name']
	tracks= playlist['tracks']
	pid=playlist['pid']

	pid_to_playlist_name[pid]=name	

	for track in tracks:
		track_uri=track['track_uri']
		#can add name and album later if needed
		if not track_uri in track_uri_to_id:
			track_uri_to_id[track_uri]=i
			pass_this_to_api=pass_this_to_api+track_uri.split(':')[2]+','
			i+=1	

		if(i==100):
			break

	if(i==100):
		break		
	


print(pass_this_to_api)
#auth key keeps changing. needs time.
headers = {'Authorization': 'Bearer BQDBTYu421OrJHTltQU8dfyaC2dmi4J3BI1uEd7T3CQPhvC2VWfKkoizokGdm_UBEf-ORG9d9mTy3FeQytx-nG0t7mMMM53ZMXzJAz11LECE3ozKt6iHHpOKUatqPLscZNSZRYe5wP7S00c8h9_o1OGvQfxhSO59bg'}

request_link='https://api.spotify.com/v1/audio-features?ids='+pass_this_to_api

r=requests.get(request_link,headers=headers)

#print(r.text)

data = r.json()
with open('data.json', 'w') as f:
    json.dump(data, f)