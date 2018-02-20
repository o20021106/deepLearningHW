from preprocess_rnn import get_data
from keras.models import Sequential
from keras.layers import Bidirectional, Masking
import h5py
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
feature_size = 39
import numpy as np

def create_model():
    
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(777, feature_size)))
    model.add(Bidirectional(LSTM(150, return_sequences = True, dropout = 0.1, kernel_initializer='normal')))
    model.add(BatchNormalization())
    model.add(LSTM(100, return_sequences = True, dropout = 0.1, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Dense(units = 100, activation = 'relu', kernel_initializer = 'normal'))
    model.add(Dropout(0.2))
    model.add(Dense(units = 100, activation = 'relu', kernel_initializer = 'normal'))
    model.add(Dropout(0.2))
    model.add(Dense(units =64, activation = 'relu', kernel_initializer = 'normal'))

    print('model created')
    return model
    
    
def train():
    X,Y,ids,x_len, y_len =  get_data()
    print(np.array(X).shape)
    model = create_model()
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.fit(X, Y, batch_size = 30, epochs=30, validation_split = 0.1)
    score = model.evaluate(X, Y, batch_size=100)
    print(score)
    model.save("model.h5")

if __name__ == "__main__":
    print('here we go!')
    train()
