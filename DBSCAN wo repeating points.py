#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 00:10:50 2020

@author: yanivamir
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 22:55:04 2020

@author: yanivamir
"""
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.spatial import distance
import random
    
    
my_data = load_iris()
data_initial = my_data["data"]



def distances_calc(data_initial):
    data = (data_initial-np.min(data_initial,0))/(np.max(data_initial,0)- np.min(data_initial,0)) 
    distances = np.ones((len(data),len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
                distances[i,j] = distance.euclidean(data[i],data[j])
    neighbohrs = distances.argsort(axis=-1)
    return(distances,neighbohrs)
            
def regionQuery(distances,neighbohrs,eps,point):
    # print(point)
    points_in_region = neighbohrs[point,:np.sum(distances[point,:] <= eps)] 
    return(points_in_region)
       
def expandCluster(points2check,cluster,distances,neighbohrs,magazine,neigh_min,eps):
    if len(magazine) == 0:
        return(cluster) 
    
    point = magazine.pop()  
    points_in_region = regionQuery(distances,neighbohrs,eps,point)
    if  len(points_in_region) >= neigh_min :

        new_points = list(set(np.setdiff1d(points_in_region,cluster).tolist()) & set(points2check))   #remove points in region already visited by other clusters
        cluster = cluster + new_points
        magazine = magazine + new_points
        # print(len(magazine))
        expandCluster(points2check,cluster,distances,neighbohrs,magazine,neigh_min,eps) 
    else:
        expandCluster(points2check,cluster,distances,neighbohrs,magazine,neigh_min,eps)        
    return(cluster)
      
def DBSCAN(initial_data,eps,neigh_min):
    distances, neighbohrs = distances_calc(initial_data)
    clusters=[]
    unclustered = []
    points2check = np.array([i for i in range(len(initial_data))])

    while len(points2check) >0: 
        # print(len(points2check))
        random_point = random.choice(points2check)
        points_in_region = regionQuery(distances,neighbohrs,eps,int(random_point))
        new_points = list(set(points_in_region) & set(points2check))  #remove points in region already visited by other clusters
        
        if  len(points_in_region) >= neigh_min & len(new_points)>1:   #len(new_points)>= neigh_min         
            cluster =  new_points
            magazine =  new_points
            final_cluster = expandCluster(points2check,cluster,distances,neighbohrs,magazine,neigh_min,eps)  
            clusters.append(final_cluster)
            points2check = np.setdiff1d(points2check,final_cluster) 

        else:
            points2check = np.setdiff1d(points2check,int(random_point))            
            unclustered.append(int(random_point))
            
    plot(clusters,unclustered,initial_data)
    return(clusters,unclustered)

def plot(clusters, unclustered, data_initial):
    for i in range(np.shape(data_initial)[1]):
        for j in range(np.shape(data_initial)[1]):
            if j > i:
                plt.xlabel(my_data.feature_names[i])
                plt.ylabel(my_data.feature_names[j])
                for cluster in clusters:
                    plt.scatter(data_initial[cluster][:,i],data_initial[cluster][:,j], alpha=0.3) 
                plt.scatter(data_initial[unclustered][:,i],data_initial[unclustered][:,j], c='black', marker = 'x' , s=50)
                plt.show()

num_of_neighbohrs = 3  # +1 is added becuase the algorithm includes the core point as its own neighbohr
epsilon = 0.5  
final_clusters, final_unclustered = DBSCAN(data_initial,epsilon,num_of_neighbohrs+1)

print('number of cluster is {}'.format(len(final_clusters)))
print('number of unclustered points is {}'.format(len(final_unclustered)))

