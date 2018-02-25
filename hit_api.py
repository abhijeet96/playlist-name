import csv
import os
import requests
import json

string_to_hit_api = ''
with open('uri_list', 'r') as myfile:
	i = 0
	for line in myfile:
		if (i>=1970202):
			line = line.replace("\r", "").replace("\n", "")
			string_to_hit_api = string_to_hit_api + line +','
			if(i%100==0 or i==2262292):
				#auth key keeps changing. needs time.
				headers = {'Authorization': 'Bearer BQBbpD5OhixzlVCBxEaaLzvzifIcpVB1Y00qKWl0uwOY51cR1ABri4NHvidCNlA7O5c9I_3r1_cABul20avrob-CgSbsi2SUUgQTcYfwOcxRd5mnovSWeONoPfGpFDG2qSermn9vToC8kYBBgS4NIT0JAp3CZmouTg'}
				string_to_hit_api = string_to_hit_api[:-1]
				#print(string_to_hit_api)
				request_link='https://api.spotify.com/v1/audio-features?ids='+ string_to_hit_api

				r=requests.get(request_link,headers=headers)

				#print(r.text)
				print(i)
				data = r.json()
				string_to_hit_api = ''
				with open(str(i) + '-' + str(i+99) +'.json', 'w') as f:
					json.dump(data, f)
		i += 1