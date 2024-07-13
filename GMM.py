#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:58:29 2020

@author: yanivamir
"""

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.stats import multivariate_normal
import random
    
    
my_data = load_iris()
data_initial = my_data["data"]


def initialize(data_initial,k):
    data = (data_initial-np.min(data_initial,0))/(np.max(data_initial,0)- np.min(data_initial,0))     
    points_list = [i for i in range(len(data))]
    random_points = random.sample(points_list,k)
    means = data[random_points]
    w = np.ones(k)/k    
    covar_3d = np.tile(np.cov(data.T,bias=True),(k,1,1))
    return(data,means,covar_3d,w)

def Expectation(data,k,means,covar_3d,w):
    prob = np.zeros((k,len(data)))
    resp = np.zeros(len(data))
    for cluster in range(k):
        # print(covar_3d[cluster,:,:])
        # print(means[cluster,:])
        prob[cluster] = multivariate_normal(means[cluster,:], covar_3d[cluster,:,:]).pdf(data) 
        # plt.contourf(data[clusters == cluster][:,0],data[clusters == cluster][:,1], rv.pdf(pos))
    resp = (w*prob.T)/np.sum(w*prob.T) 
    # likelihood = np.sum(np.log(resp))  
    clusters = np.argsort(resp,axis=1)[:,-1]
    likelihood = np.log(np.sum(np.sort(resp,axis=1)[:,-1]))

    populations = np.ones(k)*-1
    for cluster in np.unique(clusters):
        populations[cluster] = np.sum(clusters == cluster) 
    # print(populations)
    return(resp, populations,clusters,likelihood)

def Maximization(data,k,means,covar_3d,populations,clusters,w):
    for cluster in range(k):
        means[cluster] = np.mean(data[clusters == cluster],axis=0)
        covar_3d[cluster,:,:] = np.cov(data[clusters == cluster].T,bias=True)
        w[cluster] = populations[cluster]/np.sum(populations)   
    # print(w)
    return(means,covar_3d,w)  

def plot(data,means,clusters,covar_3d):
    for i in range(np.shape(data)[1]):
        for j in range(np.shape(data)[1]):
            if j > i:
                plt.xlabel(my_data.feature_names[i])
                plt.ylabel(my_data.feature_names[j])
                for cluster in np.unique(clusters):
                    plt.scatter(data[clusters == cluster][:,i],data[clusters == cluster][:,j]) 
                    # prob = multivariate_normal(means[cluster,:], covar_3d[cluster,:,:]).pdf(data)
                    # plt.contourf(data[clusters == cluster][:,0],data[clusters == cluster][:,1], np.meshgrid(prob,prob)
                plt.scatter(means[:,i],means[:,j], c='black', marker = 'x' , s=50)
                # plt.scatter(data_initial[unclustered][:,i],data_initial[unclustered][:,j], c='black', marker = 'x' , s=50)
                plt.show()
                
def fit(data_initial,k):
    data, means, covar_3d, weights =  initialize(data_initial,k)
    new_likelihood = 1000
    delta = 1
    iterations = 0
    while  (delta>0.0000000001) & (iterations<20) :
        likelihood = new_likelihood
        responsibilities, populations,clusters, new_likelihood = Expectation(data,k,means,covar_3d,weights)
        means, covar_3d, weights = Maximization(data,k,means,covar_3d,populations,clusters,weights)
        # print(new_likelihood)
        delta = np.sqrt((likelihood - new_likelihood)**2)
        # print(delta)
        iterations+=1
        # plot(data,means,clusters,covar_3d)
    return(data,means,covar_3d,clusters,responsibilities,likelihood)

k=3
likelihood_min = -np.inf
for i in range(30):  
    data,means,covariance,clusters,responsibilities,likelihood = fit(data_initial,k)
    if likelihood > likelihood_min:
        means_min = means
        covariance_min = covariance
        clusters_min = clusters
        resp_min = responsibilities
        likelihood_min = likelihood
        print(likelihood_min)
plot(data,means_min,clusters_min,covariance_min)    


