#%%
# from IPython import get_ipython
#from IPython.display import display, clear_output
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from pickle import load, dump
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

#%% Loading data

X = pd.read_csv('X',index_col=0)
Y = pd.read_csv('Y',index_col=0)

#%% scaling Y data
scaler = StandardScaler()
Y_scaled = scaler.fit_transform(Y)
dump(scaler, open('scaler.pkl', 'wb'))
scaler = load(open('scaler.pkl', 'rb'))



#%% define the keras model
model = Sequential()
model.add(Dense(444, input_dim=111, activation='relu'))
model.add(Dense(8000, activation='relu'))
model.add(Dense(444, activation='relu'))
model.add(Dense(111, activation='tanh'))

#%%

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y_scaled, epochs=500, batch_size=150, workers=8, use_multiprocessing=True)

#%% evaluate the keras model
_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))

#%% Save model
dump(model, open('metamodel.pkl', 'wb'))
print('model saved to disk')

