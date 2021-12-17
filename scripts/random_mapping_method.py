'random_mapping_method.py'

__date__='20210514'
__author__='Xu Liu'
__email__='liuxu1@sia.cn'

import numpy as np
from scipy import linalg
import time

class random_mapping_method():

    '''
    pamameters:
    targetDimen: int, the dimension of the target features
    actiFunc: string, the activation function defined within the class as follows
    scaleRate: float: the scale of the random weights 
    '''
    def __init__(self,targetDimen=100, actiFunc='sin', scaleRate=1):
        self.actiFunc = actiFunc
        self.targetDimen = targetDimen
        self.scaleRate = scaleRate
    
    def feature_mapping(self,dataSet):

        initial_dim = np.size(dataSet, 1)
        self.randomWeights = (np.random.rand(initial_dim, self.targetDimen)*2-1)*self.scaleRate
        self.randomBias = (np.random.rand(1, self.targetDimen)*2-1)*self.scaleRate
        #activation functions, not limited to the followings
        def sigmoid(dataSet):
            return 1.0 / (1 + np.exp(-dataSet))
        def sin(dataSet):
            return np.sin(dataSet)
        def linear(dataSet):
            return dataSet
        def tanh(dataSet):
            return np.tanh(dataSet)
            
        actiFun = {'sig':sigmoid, 'sin':sin, 'linear':linear, 'tanh':tanh}
        
        randomSetTemp = np.dot(dataSet, self.randomWeights) + np.tile(self.randomBias, (len(dataSet), 1))
        randomSet = actiFun[self.actiFunc](randomSetTemp)
        return randomSet

    '''
    Compute least-squares solution of the linear regression model, and other method can also be used.
    parameters:
    X: Training data, array-like of shape (n_samples, n_features). In the context of rmm, it will be the generated randomSet.
    Y: Target values, array-like of shape (n_samples,)
    '''
    def fit(self, X, Y):
        X = np.c_[X, np.ones(len(X))]  # Augment features to yield intercept
        self.coef_, self._residues, self.rank_, self.singular_ = linalg.lstsq(X, Y)

    '''
    Predict the targets using the fitted linear model
    parameter:
    X2pedict: Test samples, array-like of shape (n_samples, n_features). 
              It must be transformed by the same random mapping with the training data.
    '''
    def predict(self, X2pedict):
        X2pedict = np.c_[X2pedict, np.ones(len(X2pedict))]
        Y_Predictd = np.matmul(X2pedict, self.coef_) 
        return Y_Predictd
    
    '''
    Return the coefficient of determination and the mean square error of the prediction.
    parameters:
    X: Test samples, array-like of shape (n_samples, n_features).
    Y: True values for X, array-like of shape (n_samples,).
    '''
    def score(self, X,Y):
        Y_Predictd = self.predict(X)
        Y_mean = np.mean(Y)
        S_tol = np.sum((Y-Y_mean)**2)
        S_reg = np.sum((Y_Predictd-Y)**2)
        R2 = 1 - S_reg/S_tol
        mse = ((Y-Y_Predictd)**2).sum() / len(Y)
        return R2, mse

def loadData(filePath):
    Data = []
    fileFormat=filePath.strip().split('.')[-1]
    fr = open(filePath)
    initialData = fr.readlines()
    fr.close()
    for element in initialData:
        lineArr = element.strip().split(',')  if fileFormat == 'csv' else element.strip().split(' ')
        Data.append([float(x) for x in lineArr])
    return np.array(Data)

planet = 'datasets\\planet.xyz'
quarry = 'datasets\\quarry.xyz'
mountain = 'datasets\\mountain.xyz'

# example
if __name__ == '__main__':
    # data loading and training number
    Data = loadData(planet)
    np.random.shuffle(Data)
    L_training = int(len(Data) * 0.7)
    X, Y = Data[:,:-1], Data[:,-1]
    # the targetDimens, actiFuncs, scaleRates for planet, quarry, and mountain are
    # (500, 'sin', 4), (800, 'sin', 1), and (800, 'sin', 0.2), respectively
    # mapping time
    rmm = random_mapping_method(targetDimen=500, actiFunc='sin', scaleRate=4)
    start_mapping_time = time.time()
    data_transformed = rmm.feature_mapping(X)
    end_mapping_time = time.time()
    time_mapping = end_mapping_time - start_mapping_time
    # training time
    X_training, Y_training, X_test, Y_test = data_transformed[:L_training,:], Y[:L_training], data_transformed[L_training:,:], Y[L_training:]
    start_training_time = time.time() 
    rmm.fit(X_training, Y_training)
    end_training_time = time.time()
    time_training = end_training_time - start_training_time
    # test time
    start_test_time = time.time() 
    rmm.predict(X_training)
    end_test_time = time.time()
    time_test = end_test_time - start_test_time
    # modeling time
    time_modeling = time_mapping + time_training
    # access time
    start_access_time = time.time() 
    Y_predicted = rmm.predict(np.r_[X_training, X_test])
    end_access_time = time.time()
    time_access = end_access_time - start_access_time
    # inference accuracy
    r2_training, mseTraining = rmm.score(X_training,Y_training)
    r2_test, mseTest = rmm.score(X_test,Y_test)
    # key results
    print('r2_training:%F, r2_test:%F, mseTraining:%f, mseTest:%f, time_training:%f, time_test:%f' \
        % (r2_training, r2_test, mseTraining, mseTest, time_training, time_test))