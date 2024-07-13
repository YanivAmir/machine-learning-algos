#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:11:14 2020

@author: yanivamir
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:15:01 2020

@author: yanivamir
"""
import numpy as np 
    
class DataSet:
    
    def __init__(self,X,y,percent):
        self.X = X
        self.y = y
        self.percent = percent
        self.train
        self.test
        
    def create_train_test(self):
        m = len(self.y)
        permutation = np.random.permutation(m)
        X = self.X[permutation]
        y = self.y[permutation]
        split_index = int(m * self.percent)
        self.train = X[:split_index,:], y[:split_index]
        self.test = X[split_index:,:], y[split_index:]
        return self.train, self.test


from scipy import stats
    
class KNN:
    
    def __init__(self,train,test,k):
        self.k = k
        self.train
        self.test
        self.classifies_test
        self.precision

    def distance(_x1,_x2):
        _square_delta=(_x1-_x2)**2
        return(np.sqrt(np.sum(_square_delta)))

    def classify_test(self):
        _m=len(self.train[0])
        _n=len(self.test[0])
        _neighbors=np.zeros((_n,m_))
        for i in range(_n):
            for j in range(_m):
                _neighbors[i,j]=distance(self.test[0][i,:],self.train[0][j,:])  #caculates the euclidean distance between every test instance and a every training instance
        _k_nearest_index=np.argsort(_neighbors,1)[:,:self.k+1]     #sorts every row from low to high and gives result in indeces   
        self.classified_test=self.train[1][_k_nearest_index][:,1:]     #translates the training set indices into their label
        self.classified_test=stats.mode(self.classified_test,1)[0]    #returns the mode label of every row
        return (np.asarray(self.classified_test).reshape(-1))   #reduce the matrix in previous line to a 1-dim vector
                        
    def calc_precision(self):
        _m=self.test[1]
        self.precision=np.zeros_(m)
        for i in range(_m):
            if self.classified_test==self.test[1]:
                self.precision[i]=1
        return(self.precision.mean())                      
             
    def KNN(percent,k,X,y):
!!        train,test=create_train_test(percent,X,y) 
        self.classified_test=classify_test()
        return()    
    