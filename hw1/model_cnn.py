from preprocess_cnn import get_data
from import_data import import_data
from keras.models import Sequential
from keras.layers import Bidirectional, Masking, MaxPool2D, Conv2D, Conv1D, Input, Flatten,Reshape, ConvLSTM2D
import h5py
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.initializers import Constant
import numpy as np
from keras import backend as K
from keras.models import Model
import tensorflow as tf
feature_size = 39
width = 1
'''
def get_data():
    import_data('train')
    return fetch('train')
'''


def create_model():
    #mask = Input(shape = (777,1))
    model = Sequential()

    model.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),padding='same', return_sequences=True, input_shape=(777, width*2+1, feature_size,1)))
    model.add(Flatten())
    model.add(Reshape((777,-1)))
    model.add(Masking(mask_value=0))

    model.add(LSTM(100, return_sequences = True, dropout = 0.1, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(LSTM(100, return_sequences = True, dropout = 0.1, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Dense(units = 128, activation = 'relu', kernel_initializer = 'normal'))
    model.add(Dropout(0.3))
    model.add(Dense(units = 128, activation = 'relu', kernel_initializer = 'normal'))
    model.add(Dropout(0.3))
    model.add(Dense(units = 48, activation = 'softmax', kernel_initializer = 'normal'))
    print('model created')
    return model
    '''
    inputs = tf.unstack(inputs)
    
    #cnn
    cnn=Conv2D(10, (3,3),padding='same', data_format='channels_last' ,activation = 'relu',bias_initializer=Constant(0.01), kernel_initializer='random_uniform')
    #cnn_outputs = cnn(inputs)
    cnn_outputs_flatten = [ Flatten()(cnn(input_cnn)) for input_cnn in inputs]
    
    #cnn_outputs=Conv2D(32, (2, 2),padding='same', activation = 'relu',bias_initializer=Constant(0.01), kernel_initializer='random_uniform')(inputs)
    #cnn_outputs_flatten = Flatten()(cnn_outputs)
    masked_cnn = tf.multiply(cnn_outputs_flatten, mask)
    
    masked = Masking(mask_value = 0)(masked_cnn)
    
    bidirectional_outputs= Bidirectional(LSTM(128, return_sequences = True, dropout = 0.1, kernel_initializer='normal')(masked))
    normalized = BatchNormalization()(bidirectional_outputs)
    lstm_outputs = LSTM(96, return_sequences = True, dropout = 0.1, kernel_initializer='normal')(normalized)
    normalized = BatchNormalization()(lstm_outputs)
    outputs = Dense(units = 256, activation = 'relu', kernel_initializer = 'normal')(normalized)
    outputs= Dropout(0.3)(outputs)
    outputs = Dense(units = 128, activation = 'relu', kernel_initializer = 'normal')(outputs)
    outputs= Dropout(0.3)(outputs)
    outputs = Dense(units = 64, activation = 'relu', kernel_initializer = 'normal')(outputs)
    outputs= Dropout(0.3)(outputs)
    outputs = Dense(units = 48, activation = 'softmax', kernel_initializer = 'normal')(outputs)
    '''


def train():
    X,Y,ids,x_len, y_len, mask_input =  get_data()
    X = np.array(X)[:,:,:,:,np.newaxis]
    print(X.shape)
    model = create_model()

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.fit(X, Y, batch_size = 50, epochs=10, validation_split = 0.1)
    score = model.evaluate(X, Y, batch_size=100)
    print(score)
    model.save("model_cRnn.h5")

if __name__ == "__main__":
    print('here we go!')
    train()
