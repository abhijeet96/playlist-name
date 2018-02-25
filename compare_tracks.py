import pandas as pd
import numpy as np
import os
import json
import sys


json_filepath = os.path.join('803.json')
json_filepath = os.path.join('998.json')

data = json.load(open(json_filepath))
data2 = json.load(open(json_filepath))

tracks1 = data['tracks']
tracks2 = data['tracks']

for track in tracks1:
	print(track['track_name'])
for track in tracks2:
	print(track['track_name'])