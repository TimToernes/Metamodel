#%%

import pypsa
#import gurobipy
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from keras.models import model_from_json
import solutions
import mga_functions
from sklearn.preprocessing import StandardScaler
from pickle import load, dump

# %% import network 

network = pypsa.Network()
network.import_from_hdf5('euro_95')
network.snapshots = network.snapshots[0:10]

#%%

scaler = load(open('scaler.pkl', 'rb'))
metamodel = load(open('model.pkl', 'rb'))
# %% import meta model 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
metamodel = model_from_json(loaded_model_json)
# load weights into new model
metamodel.load_weights("model.h5")
metamodel.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
Y = pd.read_csv('test',index_col=0)
print("Loaded model from disk")


#%% Transformation matrix
T = np.array([(network.generators.type == 'wind').values.astype(int),
                (network.generators.type == 'solar').values.astype(int),
                (network.generators.type == 'ocgt').values.astype(int)])

# %%

direction = mga_functions.rand_split(111)

#%% 

def eval_model(directions,network,options,sol):
    for direction in directions:
        network,stat = mga_functions.solve(network,options,direction)
        sol.put(network)
    sol.merge()
    return sol

# %%

def eval_meta(directions,network,options,sol):
    for direction in directions:
        y2 = metamodel.predict(np.array([direction]))
        y2 = scaler.inverse_transform(y2)
        y2 = pd.Series(y2[0],index=network.generators.index)
        network.generators.p_nom_opt = y2
        sol.put(network)
    sol.merge()
    return sol 


# %%
options = dict(mga_variables=network.generators.index.values,
                mga_slack_type='percent',
                mga_slack=0.1,cpus=4)

network = mga_functions.inital_solution(network,options)
dim=111
directions = np.concatenate([np.diag(np.ones(dim)),-np.diag(np.ones(dim))],axis=0)
directions = directions[0:50]
sol1 = solutions.solutions(network)
sol2 = solutions.solutions(network)


# %%

sol2 = eval_meta(directions,network,options,sol2)

# %%

sol1 = eval_model(directions,network,options,sol1)

# %%
