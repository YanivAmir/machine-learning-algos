#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 13:21:44 2021

@author: yanivamir
"""


from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import pickle
import collections

class SVM():
    
   def  __init__(self,data,train_labels,alpha=0.1,epoch=1000):
       self.data= data
       self.train_labels = train_labels
       self.alpha = alpha
       self.w = np.reshape(np.ones(np.shape(self.data)[1])/np.shape(self.data)[1],(np.shape(self.data)[1],1))
       self.b = 1
       self.epoch = epoch
       
                      
   def hinge_loss(self):   
       h = self.w * self.data + self.b
       return(np.sum(max(0,1-self.train_labels * h)))
   
   def derivative(self,x,hypo,label):
       if label * np.sum(hypo) < 1:
           return(- label * x)
       else:
           return(0)

   def fit(self):
       for i in range(self.epoch):
           loss=[]
           sigma = 0
           for j in range(len(self.data)):
               sigma += self.derivative(self.data[j],self.w.T*self.data[j]+self.b,self.train_labels[j])               
           self.w += self.alpha * sigma
           self.b = (max(self.w*self.data[self.train_labels == -1]) + min(self.w*self.data[self.train_labels == 1]))/2
           loss = loss.append(self.loss())
       return(loss)
        
   def accuracy(self,test_x, test_labels):
        calc_h = self.w * test_x + self.b
        calc_label = [1 if calc_h >= 0 else -1]
        return(np.sum(calc_label == test_labels) / len(test_labels))
    
if __name__ == '__main__':
   
   
   all_data = pd.read_csv('/Users/yanivamir/Documents/Machine Learning/class of june 21/HW/HW7 Naive Bayes/diabetes.csv')
   data_np = np.array(all_data)
   labels = data_np[:,-1]
   all_X = np.delete(data_np,-1,1)
   features = all_data.columns
   x_train,x_test,y_train,y_test = train_test_split(all_X,labels,random_state=42)  
   
   my_SVM = SVM(x_train,y_train,0.1,1000)
   all_loss = my_SVM.fit()
   final_accuracy = my_SVM.accuracy(x_test,y_test)
   print(all_loss)
   print(final_accuracy)
   
    
        
        