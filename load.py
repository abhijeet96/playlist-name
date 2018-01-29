import pandas as pd
import numpy as np
import os
import json
import sys
print(sys.path)
import re
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from pprint import pprint

json_filepath = os.path.join('..','..', 'dataset', 'data','mpd.slice.8000-8999.json')

data = json.load(open(json_filepath))

playlists= data['playlists']
stop_words = set(stopwords.words('english'))

for playlist in playlists:
	synonyms = []
	name = playlist['name']
	letters_only = re.sub("[^a-zA-Z0-9 ]", "", name)
	lower_case = letters_only.lower().split()
	meaningful_words = [w for w in lower_case if not w in stop_words]
	final_name = " ".join( meaningful_words )
	final_stemmed = " ".join(stemmed_words)
	# for syn in wordnet.synsets(meaningful_words[0]):
	# 	for l in syn.lemmas():
	# 		synonyms.append(l.name())
	print(final_name,final_stemmed)



