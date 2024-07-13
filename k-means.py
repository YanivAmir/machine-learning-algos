#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 22:59:07 2020

@author: yanivamir
"""

import numpy as np 
import collections
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

my_iris = load_iris()
data = my_iris["data"]

#removing outliers is a good idea, alternatively running kmeans a few times and picking liwest loss is preferable
# saving data as arrays : a 1-d array that each point in the pointarray has equivalent label in the label array, with matching indices (see class solution)
# compare final result for K=3 with actual labels from data (data.target)

def clustering(X,mus,K):
    groups = collections.defaultdict(list)
    distances = np.zeros([len(X),len(mus)])  # np. zeros_like(mus) will give a matrix of zeros w/ similar dims to mus
    i=0
    for instance in X:    # in class: better use enumerate function in the for loop, counts i and instances: for i,instance in X
        j=0
        for mu in mus:
            distances[i,j]=np.linalg.norm(mu-instance)        #we can use distance.euclidean(point,mean) instead of linalg.norm  
            j+=1            
        # print(distances[i])
        # print(np.argmin(distances[i]))
        groups[np.argmin(distances[i])].append(instance)    #better to compare distances as we go, i.e check every new value if it is smaller (same as minimal J below), it is less computational-consuming    
        i+=1
    return (groups)

def plot_clusters(groups,a,b):
    colors=['r','b','g','c','m','y','b','teal','plum','lime','navy']
    for key in groups:
        plt.scatter(np.array(groups[key])[:,a],np.array(groups[key])[:,b], c=colors[key])    # no need to specifiy color because of the loop each group will recieve different color
    plt.show()
    return

def new_means(groups,K,data):
    mus=np.zeros([K,data.shape[1]])
    # print(mus)
    for group in range(K):
        mus[group,:]=np.array(groups[group]).mean(axis=0)  
    return(mus)

def plot_centers(mus,a,b):
    for i in range(len(mus)):
        plt.scatter(mus[i,a],mus[i,b], facecolors='none', marker='o',s=100,edgecolor='k')
        plt.scatter(mus[i,a],mus[i,b], marker='x',s=100,color='k')
    return
            
def converge(mu,new_mu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in new_mu]))
            
def k_means(data,K,x_axis,y_axis):
    centers = data[np.random.choice(range(len(data)),K,replace=False)]
    # plot_centers(centers,x_axis,y_axis)
    # print(centers)
    clusters = clustering(data,centers,K)
    # plot_clusters(clusters,x_axis,y_axis)
    new_centers = new_means(clusters,K,data)
    while not converge(centers,new_centers):
            centers = new_centers
            # plot_centers(centers,x_axis,y_axis)
            clusters = clustering(data,centers,K)
            # plot_clusters(clusters,x_axis,y_axis)
            new_centers = new_means(clusters,K,data)
    loss=calc_loss(clusters,centers)
    return(loss,centers,clusters) 

def calc_loss(groups,mus):
    loss=0
    for key in groups:
        points=np.array(groups[key])
        for point in points:
            loss+=(np.linalg.norm(mus[key]-point))**2
    return(loss)

col_x=0
col_y=3
max_K=9

losses=[]
for K in range(max_K):
    K+=1
    final_J,final_centers,final_clusters=k_means(data,K,col_x,col_y)
    for i in range(10)  :     
        new_J,new_centers,new_clusters=k_means(data,K,col_x,col_y) 
        if new_J < final_J:
            final_J=new_J
            final_centers=new_centers
            final_clusters=new_clusters
        
    print('for K = {} final loss is {}'.format(K,final_J))
    plot_centers(final_centers,col_x,col_y)
    plot_clusters(final_clusters,col_x,col_y)
    losses.append(final_J)
    
plt.plot(np.linspace(1,max_K,max_K),losses,'-o',c='r')
plt.show()
    
