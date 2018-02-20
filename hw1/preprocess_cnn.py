import pandas as pd
import numpy as np
import re
import os
import pickle

train_feature_path = 'mfcc/train.ark'
test_feature_path = 'mfcc/test.ark'
train_label_path = 'train.lab'
feature_size = 39
phone_letter_path = '48phone_char.map'
to_39_path = 'phones/48_39.map'
X_merge_path = 'merge.pickle'
merge_test = 'merge_test.pickle'
num_classes = 48
width = 1


def get_data():
    if not os.path.isfile(X_merge_path):   
        with open(train_feature_path, 'r') as f:
            lines = f.read().splitlines()
            ID = []
            data = np.zeros([len(lines), feature_size])
            for index, line in enumerate(lines):
                if (index%10000 == 0):
                    print("read in line "+str(index))
                line = line.split(" ")
                ID.append(line[0])
                data[index,:] = [float(i) for i in line[1:]] 
            
            data = pd.DataFrame(data)
            id_data = [i.split("_")for i in ID]
            id_data = pd.DataFrame(id_data, columns = ["speaker", "sentence", "frame"])
            id_data["frame"] = pd.to_numeric(id_data["frame"])
            data = pd.concat([data, id_data], axis=1)
            data = data.set_index(["speaker", "sentence", "frame"])
            print('finished reading pandas X')
            
        with open(train_label_path, "r") as f:
            lines = f.read().splitlines()
            label = [re.split("_|,",line) for line in lines]
            label= pd.DataFrame(label, columns = ["speaker", "sentence", "frame", "label"])
            label["frame"] = pd.to_numeric(label["frame"]) 
            label = label.set_index(["speaker", "sentence", "frame"])
            print('finished reading pandas labels')
        
        data = pd.concat([data, label], axis=1)
        grouped = data.groupby(level = ["speaker","sentence"])
        sentence_ids = list(grouped.groups.keys())
        merge_ids = []
        X_frames = []
        Y_frames = []
        
        print('data transformation starting')
        for num, i in enumerate(sentence_ids):
            if num%100 == 0:
                print('line: '+str(num))
            sentence = data.loc[i[0],i[1]]
            num_rows = sentence.shape[0]
            x_concat = []
            y_concat = []
      
            print('reading line: '+ '_'.join(i))
            for index in range(num_rows):

                x_concat.append(list(data.loc[i[0],i[1],index+1][0:feature_size].astype(np.float32)))
                y_concat.append(data.loc[i[0],i[1],index+1][feature_size])
          
            merge_ids.append('_'.join(i))
            X_frames.append(x_concat)
            Y_frames.append(y_concat)
        
        X_length = [len(i) for i in X_frames]
        Y_length = [len(i) for i in Y_frames]
        print('finished merging data')
        with open('merge.pickle', 'wb') as f:
            pickle.dump((X_frames, Y_frames, merge_ids, X_length, Y_length),f)

    print('loading data')
    with open(X_merge_path, 'rb') as f:
        X_frames, Y_frames, merge_ids, X_length, Y_length = pickle.load(f)
    with open(to_39_path, 'r') as f:
        lines = f.read().splitlines()
        map_to_39 = {}
        phone_39_int = {}
        
        for index, line in enumerate(lines):
            label = re.split('\t',line)
            if not label[1] in phone_39_int.keys():
                phone_39_int[label[1]] = len(phone_39_int)
            map_to_39[label[0]] = label[1]
        print('phone_39_int length: '+str(len(phone_39_int)))
        #input('press enter to continue')


    X = []
    Y = []
    mask = []
    print('starts padding')
    print(phone_39_int)

    for x, y, length in zip(X_frames, Y_frames, X_length):
        mask_pad = [0 for i in range(max(X_length))]
        mask_pad[0:length] = [1]*length
        x_pad = [[0.]*feature_size for i in range(max(X_length))]
        y_pad = [[0]*num_classes for i in range(max(X_length))]
        x_pad[0:length] = x
        #print(x_pad) 
        y_pad_temp = [ 0 for i in range(max(X_length))]
        y_pad_temp[0:length] = [phone_39_int[map_to_39[i]] for i in y]
        for index, num in enumerate(y_pad_temp):
            y_pad[index][num] = 1
        X.append(x_pad)
        Y.append(y_pad)
        mask.append(mask_pad)
        
        
    for i in range(len(X)):            
        for _ in range(width):
            X[i].insert(0, [0]*feature_size)
            X[i].append([0]*feature_size)
    x = []
    for index, sentence in enumerate(X):
        x_conv = []
        for j in range((width),(width+max(X_length)),1):        
            x_conv.append(X[index][j-width:j+width+1])
        x.append(x_conv)

    return(x,Y,merge_ids, X_length, Y_length, mask)

    
def get_data_test():
    if not os.path.isfile(merge_test):   
        with open(test_feature_path, 'r') as f:
            lines = f.read().splitlines()
            ID = []
            data = np.zeros([len(lines), feature_size])
            for index, line in enumerate(lines):
                #if (index%10000 == 0):
                    #print("read in line "+str(index))
                line = line.split(" ")
                ID.append(line[0])
                data[index,:] = [float(i) for i in line[1:]] 
            
            data = pd.DataFrame(data)
            id_data = [i.split("_")for i in ID]
            id_data = pd.DataFrame(id_data, columns = ["speaker", "sentence", "frame"])
            id_data["frame"] = pd.to_numeric(id_data["frame"])
            data = pd.concat([data, id_data], axis=1)
            data = data.set_index(["speaker", "sentence", "frame"])
            print('finished reading pandas X')
        '''    
        with open(train_label_path, "r") as f:
            lines = f.read().splitlines()
            label = [re.split("_|,",line) for line in lines]
            label= pd.DataFrame(label, columns = ["speaker", "sentence", "frame", "label"])
            label["frame"] = pd.to_numeric(label["frame"]) 
            label = label.set_index(["speaker", "sentence", "frame"])
            print('finished reading pandas labels')
        '''
        
        #data = pd.concat([data, label], axis=1)
        grouped = data.groupby(level = ["speaker","sentence"])
        sentence_ids = list(grouped.groups.keys())
        merge_ids = []
        X_frames = []
        
        print('data transformation starting')
        for num, i in enumerate(sentence_ids):
            if num%100 == 0:
                print('line: '+str(num))
            sentence = data.loc[i[0],i[1]]
            num_rows = sentence.shape[0]
            x_concat = []
      
            #print('reading line: '+ '_'.join(i))
            for index in range(num_rows):
                x_concat.append(list(data.loc[i[0],i[1],index+1][0:feature_size].astype(np.float32)))
          
            merge_ids.append('_'.join(i))
            X_frames.append(x_concat)
        
        X_length = [len(i) for i in X_frames]
        print('finished merging data')
        with open(merge_test, 'wb') as f:
            pickle.dump((X_frames, merge_ids, X_length),f)

    print('loading data')
    with open(merge_test, 'rb') as f:
        X_frames, merge_ids, X_length = pickle.load(f)
    with open(to_39_path, 'r') as f:
        lines = f.read().splitlines()
        map_to_39 = {}
        phone_39_int = {}
        
        for index, line in enumerate(lines):
            label = re.split('\t',line)
            if not label[1] in phone_39_int.keys():
                phone_39_int[label[1]] = len(phone_39_int)
            map_to_39[label[0]] = label[1]
        print('phone_39_int length: '+str(len(phone_39_int)))
        #input('press enter to continue')


    X = []
    mask = []

    print('starts padding')
    print(phone_39_int)

    for index, x in enumerate(X_frames):
        mask_pad = [0 for i in range(max(X_length))]
        mask_pad[0:X_length[index]] = [1]*X_length[index]
    #for x, y, length in zip(X_frames, Y_frames, X_length):
        x_pad = [[0.]*feature_size for i in range(777)]
        #y_pad = [[0]*num_classes for i in range(max(X_length))]
        x_pad[0:X_length[index]] = x
        #print(x_pad) 
        #y_pad_temp = [ 0 for i in range(max(X_length))]
        #y_pad_temp[0:length] = [phone_39_int[map_to_39[i]] for i in y]
        #for index, num in enumerate(y_pad_temp):
        #    y_pad[index][num] = 1
        X.append(x_pad)
        mask.append(mask_pad)
        
    for i in range(len(X)):            
        for _ in range(width):
            X[i].insert(0, [0]*feature_size)
            X[i].append([0]*feature_size)
    x = []
    for index, sentence in enumerate(X):
        x_conv = []
        for j in range((width),(width+777),1):        
            x_conv.append(X[index][j-width:j+width+1])
        x.append(x_conv)

    return(x,merge_ids, X_length, mask)

if __name__ == "__main__":
    X, Y, merge_ids, X_length, Y_length = get_data()
    print('x[0][0]')
    print(X[0][0])
    print(Y[0])
    print(len(X))
    print(len(Y))
    print(merge_ids[0])
    print(X_length[0])
    print(Y_length[0])    
    print(max(Y_length))
