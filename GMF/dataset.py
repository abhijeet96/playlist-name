
# coding: utf-8

# In[14]:


'''
Processing datasets. 
@author: Avi Jain
Based off NCF
'''
import scipy.sparse as sp
import numpy as np
import pandas as pd
import os
import json
import sys



class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path)
        self.playlistMatrix = self.get_playlist_dict(path)
        self.num_playlists, self.num_tracks = self.trainMatrix.shape
        
    def get_playlist_dict(self,filename):
        
        json_filepath = os.path.join(filename)
        data = json.load(open(json_filepath))
        i = 0
        j = 0
        
        playlists = data['playlists']
        playlist_matrix = {}
        pid_to_playlist_name = {}
        track_uri_to_id = {}
        id_to_track_uri = {}
        
        for playlist in playlists:
            name = playlist['name']
            tracks = playlist['tracks']
            pid = playlist['pid']
            pid_to_playlist_name[pid] = name

            for track in tracks:
                track_uri = track['track_uri']
                #can add name and album later if needed
                if not track_uri in track_uri_to_id:
                    track_uri_to_id[track_uri]=i
                    id_to_track_uri[i]=track_uri
                    i+=1
                playlist_matrix[(j,track_uri_to_id[track_uri])] = 1.0
            j+=1
        return playlist_matrix
    
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            negatives = []
            for x in arr[1: ]:
                negatives.append(int(x))
            negativeList.append(negatives)
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read file and Return dok matrix.
        '''
        json_filepath = os.path.join(filename)
        data = json.load(open(json_filepath))
        playlists= data['playlists']
        # Get number of users and items - hardcoded for now
        num_playlists, num_tracks = 1000, 34443
        track_uri_to_id = {}
        id_to_track_uri = {}
        pid_to_playlist_name = {}
        
        i = 0
        j = 0
        
        # Construct matrix
        mat = sp.dok_matrix((num_playlists+1, num_tracks+1), dtype=np.float32)
        for playlist in playlists:
            name = playlist['name']
            tracks = playlist['tracks']
            pid = playlist['pid']

            pid_to_playlist_name[pid] = name

            track_uri_list=[]

            for track in tracks:
                track_uri = track['track_uri']
                #can add name and album later if needed
                if not track_uri in track_uri_to_id:
                    track_uri_to_id[track_uri]=i
                    id_to_track_uri[i]=track_uri
                    i+=1

                playlist_id, track_id = j,track_uri_to_id[track_uri]
                mat[playlist_id, track_id] = 1.0 
            j+=1
        return mat
    


# In[15]:


dataset = Dataset('mpd.slice.0-999.json')
train = dataset.trainMatrix
mat = dataset.playlistMatrix
num_users, num_items = train.shape


# In[19]:


print(mat[(0,2)])
print(len(mat))
print(num_users,num_items)

