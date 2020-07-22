#%%
import pypsa
#import gurobipy
import numpy as np
import pandas as pd
#from scipy.spatial import ConvexHull
from keras.models import model_from_json
import solutions
import mga_functions
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso, MultiTaskLasso
from sklearn.metrics import accuracy_score, mean_squared_error
#import sklearn.metrics as sm
from pickle import load, dump
from keras.models import Sequential, load_model
from keras.layers import Dense
import model


#%% function definitions 

def init_members(n_members):
    members = []
    for i in range(n_members):
        # create model
        model = Sequential()
        model.add(Dense(444, input_dim=111, activation='relu'))
        model.add(Dense(800, activation='relu'))
        model.add(Dense(444, activation='relu'))
        model.add(Dense(111, activation='tanh'))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
        # save model
        filename = 'models/model_' + str(i + 1) + '.h5'
        model.save(filename)
        print('>Saved %s' % filename)
        members.append(model)
    return members

def fit_members(members,trainX,trainY,epochs=50,verbose=0):
    #members = load_all_members(n_members)
    #members = []
    print('Loaded %d models' % len(members))
    members_fitted = []
    for i,member in enumerate(members):
        print("training member # {}".format(i))
        member.fit(trainX, trainY,verbose=verbose, epochs=epochs, batch_size=150, workers=4, use_multiprocessing=True)
        #members.append(model)
        member_mse([member],trainX,trainY)
        members_fitted.append(member)

    member_mse(members_fitted,trainX,trainY)
    
    return members_fitted

def save_all_members(members): 
    for i,model in enumerate(members):
        filename = 'models/model_' + str(i + 1) + '.h5'
        model.save(filename)
        print('>Saved %s' % filename)

def load_all_members(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'models/model_' + str(i + 1) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models


# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
		# make prediction
		yhat = model.predict(inputX, verbose=0)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat
		else:
			stackX = np.dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX

# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members,inputX,inputY):
    stackedX = stacked_dataset(members,inputX)
    model = Lasso(alpha=0.1)
    #model = MultiTaskLasso(alpha=1)
    model.fit(stackedX,inputY)
    return model 

# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	yhat = model.predict(stackedX)
	return yhat

def member_mse(members,testX,testY):
    for model in members: 
        yhat = model.predict(testX)
        print("mse score =", mean_squared_error(testY, yhat,multioutput="uniform_average"))

def deviating_sample_idx(members,inputX,n_samples):
    yhats = []
    for model in members: 
        yhats.append(model.predict(inputX))

    stds = []
    for i_sample in range(inputX.shape[0]):
        std_value = []
        for i_value in range(inputX.shape[1]):
            member_val = []
            for i_member in range(len(members)):
                member_val.append(yhats[i_member][i_sample,i_value])
            std_value.append(np.std(member_val))
        stds.append(np.mean(std_value))

    sorted_stds = stds.copy()
    sorted_stds.sort()

    idx = [stds.index(val) for val in sorted_stds[-n_samples:]]
    return idx
#%% Loading data
n_pool = 1000
poolX = np.array([model.rand_split(111)])
for i in range(n_pool-1):
    poolX = np.append(poolX,[model.rand_split(111)],axis=0) 

X = pd.read_csv('X',index_col=0)
Y = pd.read_csv('Y',index_col=0)

# scaling Y data
scaler = StandardScaler()
Y_scaled = scaler.fit_transform(Y)
dump(scaler, open('scaler.pkl', 'wb'))
scaler = load(open('scaler.pkl', 'rb'))

msk = np.random.rand(len(X)) < 0.8
trainX = X[msk]
trainX = trainX.reset_index(drop=True)
testX = X[~msk]
testX = testX.reset_index(drop=True)
trainY = Y_scaled[msk]
testY = Y_scaled[~msk]
#%% Creating sub regression models 

members = init_members(5)

#%%
for i in range(5):
    # %% Fit sub regression models 
    members = load_all_members(n_models=5)
    members = fit_members(members,trainX,trainY,epochs=50,verbose=0)
    save_all_members(members)
    # %% Accuracy of individual models 
    member_mse(members,testX,testY)

    # %% fit stacked model using the ensemble
    stacked_model = fit_stacked_model(members, trainX, trainY)

    # %% Accucarcy of stacked model 
    yhat = stacked_prediction(members, stacked_model, testX)
    print("mse score =",mean_squared_error(testY, yhat,multioutput="uniform_average"))

    # %% find new points to add to training set

    new_idx = deviating_sample_idx(members,poolX,20)
    newX = poolX[new_idx]

    # %% intialize network model

    network = model.import_network(10)
    network = model.initialize_network(network)

    # %% compute new Y
    newY = []
    for X in newX:
        Y = model.evaluate(network,X)
        newY.append(Y)

    newY_scaled = scaler.transform(newY)

    # %% add new data to training set
    for X in newX:
        trainX.loc[len(trainX)+1] = X

    trainY = np.concatenate([trainY,newY_scaled])

# %%
