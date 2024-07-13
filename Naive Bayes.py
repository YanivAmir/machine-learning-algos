#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 12:50:44 2021

@author: yanivamir
"""

import pandas as pd
import numpy as np

class DataSet():
    
    def __init__(self, X,Y):
        assert len(X)==len(Y),'labels should be same length as data' # sanity check
        self.X=X
        self.Y=Y
        
    def train_test_random_split(self,p_train):
        """ returns X_train, Y_train, X_test, Y_test according to percentaget
        defined in p_train after randomly permuting the data """
        assert 0<p_train<=1
        data_len=len(self.X)
        train_len=round(p_train*data_len)
        perm= np.random.permutation(data_len)
        train_pos=perm[:train_len]
        test_pos=perm[train_len:]
        return self.X[train_pos],self.Y[train_pos],self.X[test_pos],self.Y[test_pos]
    
class NaiveBayes():
    
    def __init__(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.means = np.zeros((len(np.unique(self.y_train)),np.shape(self.X_train)[1]))
        self.vars = np.zeros((len(np.unique(self.y_train)),np.shape(self.X_train)[1]))
        
    def mean(self,label):
        return((self.X_train[self.y_train==label].mean(axis=0)))
   
    def var(self,label):
        return((self.X_train[self.y_train==label].std(axis=0))**2)
        # return(np.sum((self.X_train[self.y_train==label] - self.means[label]) **2,axis=0) / (sum(self.y_train==label)-1) )  #this is the problem
    
    def train(self):
        for label in range(len(np.unique(self.y_train))):
            self.means[label] = NaiveBayes.mean(label)
            self.vars[label] = NaiveBayes.var(label) 
        return()
    
    def prior(self):
        un,nums = np.unique(self.y_train,return_counts=True)
        prior = np.zeros(len(un))
        for label in range(len(un)):
            prior[label]=nums[label]/len(self.y_train)
        return(prior)
    
    def calc_numerator(self, v,u, var):
        n = -(v-u)**2
        d = 2*var
        return np.exp(n/d)    
            
    def predict(self,X_test):   
        l = len(np.unique(self.y_train))
        n,m =  np.shape(X_test)                  
        predictions = np.zeros((l,n,m))
        for label in range(l):           
            numerator = NaiveBayes.calc_numerator(X_test, self.means[label], self.vars[label])
            denominator = np.sqrt(2*np.pi*self.vars[label])
            predictions[label] = numerator / denominator
        return(predictions)
    
    def accuracy(self,X_test,y_test):
        predictions = NaiveBayes.predict(X_test)
        prior = NaiveBayes.prior()
        l = len(np.unique(self.y_train))
        likelihood = np.zeros((len(X_test),l))
        for label in range(l):
            likelihood[:,label]=prior[label]*np.product(predictions[label,:,:],axis=1) 
        calculated = np.argmax(likelihood,axis=1)
        return(np.mean(calculated==y_test),calculated)
    
    def precision(self,calculated,y_test):
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0
        for i in range(len(y_test)):
            if calculated[i] == y_test[i]:
                if y_test[i] == 0:
                    true_pos+=1
                else:
                    true_neg+=1
            else:
                if y_test[i] == 0:
                    false_pos +=1
                else:
                    false_neg +=1
        n= len(y_test)
        print('false pos = {:.2%}'.format(false_pos/n))  
        print('false neg = {:.2%}'.format(false_neg/n))     
        print('true pos = {:.2%}'.format(true_pos/n))
        print('true neg = {:.2%}'.format(true_neg/n))
            
           
if __name__=='__main__':
    
    all_data = pd.read_csv('/Users/yanivamir/Documents/Machine Learning/class of june 21/HW/HW7 Naive Bayes/diabetes.csv')
    data_np = np.array(all_data)
    labels = data_np[:,-1]
    all_X = np.delete(data_np,-1,1)
    features = all_data.columns
    
    print(__file__)
    print(__name__)
    
    data_set = DataSet(all_X,labels)
    X_train,y_train,X_test,y_test=data_set.train_test_random_split(0.8)
    NaiveBayes=NaiveBayes(X_train,y_train)
    NaiveBayes.train()
    accuracy,calculated_labels = NaiveBayes.accuracy(X_test,y_test)
    print('Success percentage {:.2%}'.format(accuracy))
    NaiveBayes.precision(calculated_labels,y_test)