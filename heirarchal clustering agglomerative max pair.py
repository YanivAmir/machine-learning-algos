#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:30:38 2020

@author: yanivamir
"""

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from scipy.spatial import distance
    
    
my_data = load_wine()
data_initial = my_data["data"]

def search_min_distance(clusters,data):
    min_distance = 10          # instead of arbitrary we can use max() to take in the largest number and compare to it
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if j > i:          # can use np.inf to insert infinity value to diagonal
                point_distances=np.zeros((len(clusters[i]),len(clusters[j])))    #option of comparing as we go as alternative to matrix, can be faster
                for a,instance_i in enumerate(clusters[i]):     #no need to screen all points again!! the largest distance between a new cluster and a different one
                # is the largest of the 2 largest distances of each of the adjoined clusters and the different one, thst is why a matrix make sense cause you have to revisit it each iteration
                    for b,instance_j in enumerate(clusters[j]):      #np.maximum is used to compare arrays and give one array of the largest number of all arrays
                        point_distances[a,b] = distance.euclidean(data[instance_i],data[instance_j])
                # print(np.max(point_distances))
                # print(min_distance)
                if np.max(point_distances) < min_distance:     #use np.unravel_index(np.argmin(point_distances), point_distances.shape) to find i,j in a single line
                    min_row = i
                    min_col = j
                    min_distance = np.max(point_distances)
    print('normalized, minimal, maximal pair-wise distance is {}'.format(min_distance))
    return(min_row,min_col,min_distance)

def clustering(clusters,data,stop,iterations,min_distance,stop_number):
    print('\niteration {} '.format(iterations))
    print('{} clusters'.format(len(clusters)))
    if stop_check(stop,len(clusters),iterations,min_distance,stop_number):
        # plot(clusters,data_initial)
        return(clusters)    
    else:
        i2cluster,j2cluster,min_distance = search_min_distance(clusters,data)
        # print(i2cluster)
        # print(j2cluster)
        clusters[i2cluster] = np.concatenate((clusters[i2cluster],clusters[j2cluster])) #can use define clusters as just a list (next code line) and update using clusters[i]=clusters[i]+clusters[j]
        del(clusters[j2cluster])  #can generate array during delete:  clusters= np.delete(clusters,j)
        # print(clusters)
        clustering(clusters,data,stop,iterations+1,min_distance,stop_number)
    return(clusters)
    
def plot(clusters,data_initial,falses):
    for i in range(np.shape(data_initial)[1]):
        for j in range(np.shape(data_initial)[1]):
            if j > i:
                plt.xlabel(my_data.feature_names[i])
                plt.ylabel(my_data.feature_names[j])
                for cluster in clusters:
                        plt.scatter(data_initial[cluster][:,0][:,i],data_initial[cluster][:,0][:,j])     #enter the axis for plotting instead of '1' and '3'
                plt.scatter(data_initial[falses][:,i],data_initial[falses][:,j], c='r', marker='x', s=50)
                plt.show()
        
def stop_check(stop,n_clusters,iterations,min_distance,stop_number):
    Condition=False
    if (stop=='clusters') and (n_clusters < stop_number+1):
        Condition=True
    if (stop=='iterations') and (iterations > stop_number+1):
        Condition=True 
    if (stop=='distance') and (min_distance > stop_number+1):
        Condition=True
    return(Condition)

def agglomerative(data_initial,stop_condition,stop_number):    #the gunction name shhould be "fit"
    data_normalized = (data_initial-np.min(data_initial,0))/(np.max(data_initial,0)- np.min(data_initial,0))        # w/ normalized data
    
    clusters = np.reshape([i for i in range(len(data_normalized))],(len(data_normalized),1))   #can use define clusters as just a list (next code line) and update using clusters[i]=clusters[i]+clusters[j]
    clusters = [clusters[i] for i in clusters]     #can use asarray? 
    
    iterations=0
    if stop_condition == 'distance':
        min_distance = stop_number - 0.1
    else:
        min_distance = 10
    final_clusters=clustering(clusters,data_normalized,stop_condition,iterations,min_distance,stop_number)
    return(final_clusters)
 
    
def accuracy(final):   #valid only for clusters = 3
    target = my_data["target"]
    final_final_score = 0
    final_falses = []
    for i in range(len(final)):
        for j in range(len(final)):
            target[target==i]=-1
            target[target==j]=i
            target[target==-1]=j
            groups = np.zeros(len(target))-1
            for cluster in range(len(final)):
                for j in range(len(final[cluster])):
                    groups[int(final[cluster][j])] = cluster
            # target[target=2]=3
            score = [target[i]==groups[i] for i in range(len(target) )]
            falses=[]
            falses.append([i for i in range(len(score)) if score[i]==0 ])
            final_score = sum(score)/len(score)
            if final_score > final_final_score:
                final_final_score=final_score
                final_falses=falses
    return(final_final_score,final_falses)


stop_condition = 'clusters'   # enter either 'distance', 'cluster' number or 'iterations'
stop_number = 3   #enter stop condition threshold

final = agglomerative(data_initial,stop_condition,stop_number)
print('\n')
score,falses = accuracy(final)
print('final score is {}'.format(score))

plot(final,data_initial,falses)




