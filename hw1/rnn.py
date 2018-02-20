import numpy as np
import re
from preprocess_rnn import get_data_test
import h5py
from keras.models import load_model
import os
import sys

data_path = sys.argv[1]
output_path = os.path.abspath(sys.argv[2])
cwd = os.getcwd()
os.chdir(data_path)
to_39_path = 'phones/48_39.map'
phone_letter_path = '48phone_char.map' 

'''to_39_path = '../data/48_39.map'
phone_letter_path = '../data/48phone_char.map'
'''
sample_path = os.path.join(cwd,'sample.csv')
model_path = os.path.join(cwd,'model.h5')

with open(to_39_path , 'r') as f:
    lines = f.read().splitlines()
    map_to_39 = {}
    phone_39_int = {}
    
    for index, line in enumerate(lines):
        label = re.split('\t',line)
        if not label[1] in phone_39_int.keys():
            phone_39_int[label[1]] = len(phone_39_int)
        map_to_39[label[0]] = label[1]
    int_39_phone ={}
    
    for key, value in phone_39_int.items():
        int_39_phone[value]= key
    print('int_39_phone length: '+str(len(phone_39_int)))
    
with open(phone_letter_path, 'r') as f:
    lines = f.read().splitlines()
    phone_letter = {}
    for line in lines:
        label = re.split('\t',line)
        phone_letter[label[0]] = label[2]

model = load_model(model_path)
X, ids, seq_len, mask = get_data_test()
predictions = model.predict(X)
predictions =  np.argmax(predictions, axis= 2)
predictions = [predictions[index, 0 : seq_l]for index, seq_l in enumerate(seq_len) ]
predictions_letter = []
for prediction in  predictions:
    phones = []
    for column in range(prediction.shape[0]):
            #print(prediction.shape)            
        phones.append(phone_letter[int_39_phone[prediction[column]]])
    predictions_letter.append(phones)
    
    
predictions_trimed = []
for prediction in predictions_letter:
    buffer_symbol = ''
    trimed = []    
    for phone in prediction:
        if phone!= buffer_symbol:
            trimed.append(phone)
            buffer_symbol = phone
    if trimed[0] == 'L':
        trimed = trimed[1:]
    if trimed[-1] == 'L':
        trimed = trimed[0:-1]
    
    predictions_trimed.append(trimed)
            
predictions_concatenate = [''.join(prediction) for prediction in predictions_trimed]

        
predictions = {}
for prediction, sen_id in zip(predictions_concatenate, ids):
    predictions[sen_id] = prediction

print('starts wrting file')
with open(sample_path, 'r') as f:
    with open(output_path, 'w') as g:
        g.write("id,phone_sequence\n")
        submit = np.loadtxt(f, skiprows = 1, dtype = str, delimiter = ',')
        for i in range(submit.shape[0]):
            g.write(','.join([submit[i,0],predictions[submit[i,0]]])+'\n')
