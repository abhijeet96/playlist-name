import gensim
from gensim.models import Word2Vec, KeyedVectors
import os
import json
import pickle
import heapq
import scipy.sparse as sp
from collections import Counter

filepath=r'\\SAPNA\Users\Sapna singh\Desktop\4-2\malap\p2v\prod_vec_model'
model = Word2Vec.load(filepath)

with open('uri_to_id.p', 'rb') as fp:
    uri_to_id= pickle.load(fp)

with open('id_to_uri.p', 'rb') as fp:
    id_to_uri= pickle.load(fp)

print(uri_to_id["6OxPxIrGaGCYJQxaKSaA4U"])

raw_filepath=r'C:\Users\Sapna singh\Desktop\4-2\malap\single'

ct=0

def jaccard(predictions_for_pid,pid):
	fl=0
	actual=[]

	print(pid)
	with open('testin.txt','r') as test:
		for test_line in test:
			test_pid=test_line.split(' ')[0]
			test_uri=test_line.split(' ')[1].strip()
			#print(test_pid)

			if(str(test_pid)==pid):
				actual.append(test_uri)
				fl=1
			elif(fl==1):
				break

	count=0

	for i in predictions_for_pid:
		for j in actual:
			#print(int(i[1]),int(uri_to_id[j]))
			# if (int(i[1])==int(uri_to_id[j])):
			# 	count+=1
			# 	break
			if (int(i)==int(uri_to_id[j])):
				count+=1
				break

	#predictions_for_pid=[int(i[1]) for i in predictions_for_pid]
	predictions_for_pid=[int(i) for i in predictions_for_pid]
	actual=[int(uri_to_id[j]) for j in actual]

	print(actual,predictions_for_pid)
	return count/10




with open('trainin.txt', 'r') as train: 

	predictions_for_pid=[]
	init_ind='0'
	training_tracks = []
	#prev_pid='0'
	for line in train:
		pid = str(line.split(' ')[0])		
		#print(pid)
		if(pid!=init_ind):
			#heapq.heapify(predictions_for_pid)
			#heap_predictions=heapq.nlargest(10,predictions_for_pid)

			#print(pid,answer)
			heap_predictions = []
			count_predictions =  Counter(predictions_for_pid)
			for k, v in count_predictions.most_common(15):
				# print(k,v)
				if k not in training_tracks:
					heap_predictions.append(k)
				else:
					print("Is already part of training_tracks")
			# print(heap_predictions)
			#its here. bug found
			jaccard_coeff=jaccard(heap_predictions,init_ind)
			init_ind=pid
			predictions_for_pid=[]		
			training_tracks = []
			print(pid,jaccard_coeff)

			ct+=1
			if(ct==150):
				break


		train_uri=line.split(' ')[1].strip()

		try:
			track_op=model.wv.most_similar(str(uri_to_id[train_uri]))
			training_tracks.append(uri_to_id[train_uri])
		except:
			pass
			#print(uri_to_id[train_uri])
		# print(training_tracks)
		for track_data in track_op:
			# if track_data[0] not in training_tracks:
			predictions_for_pid.append(track_data[0])
			# predictions_for_pid.append((track_data[1],track_data[0]))

		#prev_pid=pid




# for filename in os.listdir(raw_filepath):
#     data = json.load(open(raw_filepath+"\\"+ filename))

    # playlists= data['playlists']

    # for playlist in playlists:
    #     name = playlist['name']
    #     tracks= playlist['tracks']
    #     pid=playlist['pid']
    #     num_tracks=playlist["num_tracks"]
# train = sp.load_npz('training_sparse_matrix.npz')
# train = sp.csr_matrix(train)
# print(train[0].split(' '))
# track_uri_list=[]

# #print(pid)
# model_op=[]

# if num_tracks>=50:
#     for track in tracks:
#         track_uri=track['track_uri'].split(':')[2]
#         track_pos=track["pos"]     
        
#     #try:
#         track_op=model.wv.most_similar(str(uri_to_id[track_uri]))                
#         track_op= [id_to_uri[int(i[0])] for i in model_op]
#         model_op.append(track_op)
#         print(model_op)         

#         if(ct==50):
#         	break
#         ct+=1

#     #except:
#         print("not in vocab")

# actual=[]
# fl=0
# print('done')
# with open('testin.txt', 'r') as test: 
#     for line in test:
#         if(str(line.split(' ')[0])==str(pid)):
#             actual.append(line.split(' ')[1].strip())
#             fl=1
#         elif (fl==1):
#             break
# positive=0
#         #print('done2')
# for predictions in model_op:
#     for real in actual:
#         if real==predictions:
#             positive+=1
#             break
# print((positive/len(model_op))*100)


        

