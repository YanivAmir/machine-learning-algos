#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 15:01:07 2021

@author: yanivamir
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:43:30 2021

@author: yanivamir
"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection 
import pickle
import collections
import warnings
warnings.filterwarnings("ignore")

class Node():
    '''Trees are generated minimizing gini index at node split'''
    '''Hyperparameters: maximum depth of tree (dpeth_max) and leaf homogeneity threshold (homogeneity_TH, above which the node becomes a leaf)'''
    
    def __init__(self,data,labels,columns_choice = None ,depth=0,depth_max=3,homogeneity_TH=0.95):
        self.data = data
        self.labels = labels
        self.leaf = None
        self.feature = None
        self.value = None
        self.depth = depth  
        self.depth_max = depth_max
        self.homogeneity_TH = homogeneity_TH
        if columns_choice.any() == None:    
            self.columns_choice= np.arange(np.shape(self.data)[1])
        else:
            self.columns_choice = columns_choice
           
    def calc_gini(self,y1,y2):
        gini=0
        total_branch = len(y1)+len(y2)
        for y in [y1,y2]:
            if len(y) == 0:
                continue
            counts = np.unique(y,return_counts=True)[1]
            if len(counts)==1:
                counts = [counts,0]
            gini += (len(y)/total_branch)*(1-(counts[0] / len(y)) **2 - (counts[1] / len(y)) **2)
        return(gini)
    
    def search_for_split(self):
        min_gini = 1
        split_feature = 0 
        split_value = 0   
        for col in self.columns_choice:
            x = self.data[:,col]
            sort_index = np.argsort(x, axis=0)
            x = self.data[sort_index,col]
            y = self.labels[sort_index]
            for i in range(len(x)-1):
                i+=1
                y_left = y[:i]
                y_right = y[i:]
                gini = self.calc_gini(y_left,y_right)
                if (gini <= min_gini) :
                    min_gini = gini
                    split_feature = col
                    split_value = x[i]
        return(split_feature,split_value)  
    
    def split(self):
        left_X = self.data[self.data[:,self.feature] >= self.value]
        left_y = self.labels[self.data[:,self.feature] >= self.value]
        right_X = self.data[self.data[:,self.feature] < self.value]
        right_y = self.labels[self.data[:,self.feature] < self.value]
        return(left_X,right_X, left_y,right_y)
        
    def stop(self):
       count = collections.Counter(self.labels)
       if self.depth == self.depth_max or len(count)==1:
           self.leaf = count
           return(True)
       else:
           return(False)
                       
    def split_node(self):           
       if self.stop():          
           return()
       else:
           self.feature, self.value = self.search_for_split()
           self.data_left,self.data_right,self.labels_left, self.labels_right = Node.split(self)                       
           self.left_node = Node(self.data_left,self.labels_left,self.columns_choice,self.depth+1)
           self.left_node.split_node()
           self.right_node = Node(self.data_right,self.labels_right,self.columns_choice,self.depth+1)
           self.right_node.split_node()
       return()
   
    def test_x(self,x):
        if self.leaf != None :
            return(self.leaf)
        else:
            if x[self.feature] >= self.value:
                return( self.left_node.test_x(x))
            else:
                return (self.right_node.test_x(x))

    def certainty(self,data,certainty,indices):
        for index in indices:
            leaf_counter = self.test_x(data[index])
            if certainty[index] == []:
                certainty[index] = leaf_counter
            else:
                certainty[index] += leaf_counter        
        return(certainty)
    
    def accuracy(self,calculated_labels,y_test):
        return(np.sum([calculated_labels[i] == y_test[i] for i in range(len(y_test)) ])/len(y_test))
    
    def print_tree(self):
        if self.leaf!=None:
            homogeneity = 100*self.leaf.most_common(1)[0][1] / sum(self.leaf.values())
            print('\t'*(self.depth),'Leaf@ depth ',self.depth,' Label ',self.leaf.most_common(1)[0][0],' Homogeneity ',round(homogeneity,3),'%')
        else:
            print('\t'*(self.depth),'Split@ feature#: ',self.feature,' @Value ',self.value,' @Depth ',self.depth)
            self.left_node.print_tree()
            self.right_node.print_tree()
    
   
class RandomForest():
    '''Forest generated through randomination of instances via bootstraping for each Tree'''
    ''' and a subset of randominzed features for each Tree (subset of log2 of the total number of features is taken)'''
    '''hyperparametes: maximum dpeth of tree (dpeth_max) and leaf homogeneity threshold (homogeneity_TH)'''
    '''as well as the maximum number of trees in the forest (num_trees)'''
    '''gini loss is used to measure forest overall loss'''   
    
    def __init__(self,train_data,train_labels,test_data,test_labels,num_trees=100,depth=0,depth_max=4,homogeneity_TH=0.95):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.depth = depth  
        self.num_trees = num_trees
        self.depth_max = depth_max
        self.homogeneity_TH = homogeneity_TH  
        self.train_results = {i: [] for i in range(len(self.train_data))} 
        self.validation_results = {i: [] for i in range(len(self.train_data))}
        self.test_results = {i: [] for i in range(len(self.test_data))} 
        self.training_loss = []
        self.validation_loss = []
        self.training_accuracy = []
        self.validation_accuracy = []
        self.tree_indices = []
        self.test_indices = np.arange(len(self.test_data))
        self.loss_min=1000
        
    def generate_forest (self)  :   
        indices = np.arange(len(self.train_data))
        for i in range(self.num_trees):
            train_indices = np.random.choice(indices,len(self.train_data),replace=True)
            validation_indices = [indices[i] for i in range(len(train_indices)) if indices[i] not in train_indices ]             
            columns_choice = np.random.choice(np.shape(self.train_data)[1],int(np.log2(np.shape(self.train_data)[1])),replace=False)       
            Tree = Node(self.train_data[train_indices],self.train_labels[train_indices],columns_choice,0)
            Tree.split_node()

            self.train_results = Tree.certainty(self.train_data,self.train_results,train_indices)   
            self.validation_results = Tree.certainty(self.train_data,self.validation_results,validation_indices)  
            # self.test_results = Tree.certainty(self.test_data,self.test_results,self.test_indices)
            
            # if i<20:
            #     print('Adding Tree #',i,'to forest')                            
            if i>=30 and i % 5 == 0:
                self.get_loss()
                self.plot_loss(i)
                print('Adding Tree #',i,'to forest') 
                if i % 100 == 0 :
                    print('Tree example #',i)
                    Tree.print_tree() 
            if i>=30 and self.validation_loss[-1] < self.loss_min:
                self.loss_min = self.validation_loss[-1]
                minimal_tree = i
                accuracy_at_minimal_loss = self.validation_accuracy[-1]
                # print('minimal validation loss reached. current test accuracy is ',self.test())
                # self.save_forest()      
        return(round(self.loss_min,2),minimal_tree,round(accuracy_at_minimal_loss,2))
    
    def plot_loss(self,index):
        self.tree_indices.append(index)
        fig, axs = plt.subplots(2,2)
        fig.tight_layout()
        axs[0,0].plot( self.tree_indices,self.validation_loss,  'tab:red')
        axs[0,0].set_title('Validation Loss')
        axs[0,1].plot( self.tree_indices,self.training_loss,'tab:blue')
        axs[0,1].set_title('Training Loss')
        axs[1,0].plot( self.tree_indices,self.validation_accuracy,  'tab:red')
        axs[1,0].set_title('Validation Accuracy')
        axs[1,0].set_ylim(60,100)
        axs[1,1].plot( self.tree_indices,self.training_accuracy,  'tab:blue')
        axs[1,1].set_title('Training Accuracy')
        axs[1,1].set_ylim(60,100)
        plt.pause(0.05)
        return
             
    def get_loss(self):   
        loss,accuracy = self.calc_loss(self.train_labels,self.train_results)
        self.training_loss.append(loss)
        self.training_accuracy.append(accuracy)        
        loss,accuracy =self.calc_loss(self.train_labels,self.validation_results)
        self.validation_loss.append(loss)
        self.validation_accuracy.append(accuracy)
        return
            
    def calc_loss(self,labels,results):     
        calculated_labels,instances_coverage,homogeneity_per_instance = np.zeros(len(labels)),np.zeros(len(labels)),np.zeros(len(labels))
        weighted_loss = 0
        accuracy = 0
        for i in range(len(labels)):
            calculated_labels[i] = results[i].most_common(1)[0][0]
            instances_coverage[i] = sum(results[i].values())
            homogeneity_per_instance[i] = results[i].most_common(1)[0][1] / sum(results[i].values())
        total_coverage = np.sum(instances_coverage)
        for i in range(len(labels)):       
            if labels[i] == calculated_labels[i]:
                weighted_loss +=  (1-homogeneity_per_instance[i]**2 )  
                accuracy += 1
            else:
                weighted_loss += (1-(1-homogeneity_per_instance[i])**2 ) 
        return(weighted_loss,100*accuracy/len(labels))
    
    #
    
    # def test(self):
    #     test_calculated_labels = self.test_results.most_common(1)[0][0]
    #     test_accuracy = [test_calculated_labels[i] == self.test_labels[i] for i in range(len(self.test_labels))].sum()
    #     return(test_accuracy)
        

    def save_forest(self,file_name = 'my_forest'):
        return(pickle.dump(file_name,self)   )  
  
    def load_tree(self,file_name = 'my_forest'):
        return (pickle.load(file_name))
 
    
if __name__ == '__main__':
    
    breast_cancer = datasets.load_breast_cancer()
    data=breast_cancer.data
    labels=np.array(breast_cancer.target)
    features = breast_cancer.feature_names  
    indices = np.arange(len(data))

    # X_train,X_test,y_train,y_test = model_selection.train_test_split(data,labels,test_size=0.2)
    train_indices = np.random.choice(indices,int(0.95*len(data)),replace=False)
    test_indices = [indices[i] for i in range(len(data)) if indices[i] not in train_indices ]   
    
    # Hyperparapmeters
    num_trees=3000
    depth_max=5
    homogeneity_threshold=0.95
    
    starting_depth = 0
       
    Breast_Cancer_Forest = RandomForest(data[train_indices],labels[train_indices],data[test_indices],labels[test_indices],num_trees,starting_depth,depth_max,homogeneity_threshold)
    minimal_validation_loss,minimal_tree,accuracy_at_minimal_loss = Breast_Cancer_Forest.generate_forest()
    print('Minimal loss for the validation set of ',minimal_validation_loss,' was reached after ',minimal_tree,' trees were added to the forest')
    print('Test set accuracy at minial test loss is ',accuracy_at_minimal_loss)


 