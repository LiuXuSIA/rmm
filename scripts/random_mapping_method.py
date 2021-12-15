'random_mapping.py'

__date__='20210514'
__author__='liuxu'
__email__='liuxu1@sia.cn'

import numpy as np
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.metrics import roc_curve,roc_auc_score, precision_score, recall_score,f1_score
from sklearn.linear_model import LogisticRegression
import pcdProcess 

import scipy.sparse as sp
from scipy import linalg
from scipy import optimize
from scipy import sparse
from scipy.special import expit
from joblib import Parallel

class random_mapping_method():
    def __init__(self,targetDimen=100,actiFunc='sin', scaleRate=1):
        self.actiFunc = actiFunc
        self.targetDimen = targetDimen
        self.scaleRate = scaleRate
    
    def feature_mapping(self,dataSet,biasEnable=True):

        initial_dim = np.size(dataSet, 1)
        self.randomWeights = (np.random.rand(initial_dim, self.targetDimen)*2-1)*self.scaleRate
        self.randomBias = (np.random.rand(1, self.targetDimen)*2-1)*self.scaleRate if biasEnable else np.zeros([1, self.targetDimen])
    
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

    def fit(self, X, Y):
        self.clf = LinearRegression(fit_intercept=False)
        self.clf.fit(X, Y)

        # X_offset = np.average(X, axis=0)
        # X_offset = X_offset.astype(X.dtype, copy=False)
        # X -= X_offset
        # X_scale = np.ones(X.shape[1], dtype=X.dtype)
        # Y_offset = np.average(y, axis=0)
        # Y = Y - Y_offset

        self.coef_, self._residues, self.rank_, self.singular_ = linalg.lstsq(X, Y)
        print(self.coef_.shape)
        # self.coef_ = self.coef_.T
        # if Y.ndim == 1:
        #     self.coef_ = np.ravel(self.coef_)
        # def _set_intercept(self, X_offset, y_offset, X_scale):
        #     """Set the intercept_"""
        #     if self.fit_intercept:
        #         self.coef_ = self.coef_ / X_scale
        #         self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)
        #     else:
        #         self.intercept_ = 0.0

    def predict(self, X2pedict):
        Y_Predictd = self.clf.predict(X2pedict)
        Y_Predictd_1 = np.matmul(X2pedict, self.coef_) 
        return Y_Predictd, Y_Predictd_1

    def score(self, X,Y,r2=False, mse=False):
        Y_Predictd, Y_Predictd_1 = self.predict(X)
        scores = []
        if r2:
            r2_value = self.clf.score(X,Y)
            scores.append(r2_value)
        if mse:
            mse_value = ((Y-Y_Predictd)**2).sum() / len(Y)
            mse_value_1 = ((Y-Y_Predictd_1)**2).sum() / len(Y)
            scores.append(mse_value_1)
        return scores[0] if len(scores)==1 else scores

rootPath = 'E:\\PC1SIA\\Study\PHD\\Work\\Research\\paperWriting\\RAL2021\\'
rootPath_AAAI = 'E:\\PC1SIA\\Study\\PHD\\Work\\Research\\paperWriting\\AAAI2022\\'
elevation_planetary_large = rootPath + 'dataSet\\elevation_planetary_large.xyz'
elevation_quarry = rootPath_AAAI + 'dataSet\\stonePit1.xyz'
elevation_quarry_small = rootPath_AAAI + 'dataSet\\elevation_quarry_small.xyz'

Data = pcdProcess.loadData(elevation_quarry_small)
np.random.shuffle(Data)
L_training = int(len(Data)*(1-0.3))
X, Y = Data[:,:-1], Data[:,-1]

clf=random_mapping_method(targetDimen=900, actiFunc='sin',scaleRate=2)
start_mapping_time = time.time()
data_transformed = clf.feature_mapping(X)
end_mapping_time = time.time()
time_mapping = end_mapping_time - start_mapping_time

X_training, Y_training, X_test, Y_test = data_transformed[:L_training,:], Y[:L_training], data_transformed[L_training:,:], Y[L_training:]
start_training_time = time.time() 
clf.fit(X_training, Y_training)
end_training_time = time.time()
time_training = end_training_time - start_training_time

start_test_time = time.time() 
clf.predict(X_training)
end_test_time = time.time()
time_test = end_test_time - start_test_time

time_modeling = time_mapping + time_training

start_access_time = time.time() 
Y_predicted = clf.predict(np.r_[X_training, X_test])
end_access_time = time.time()
time_access = end_access_time - start_access_time

mseTraining, mseTest =  clf.score(X_training,Y_training,mse=True), clf.score(X_test,Y_test,mse=True)
print('mseTraining:%f, mseTest:%f, time_training:%f, time_test:%f' % (mseTraining, mseTest, time_training, time_test))