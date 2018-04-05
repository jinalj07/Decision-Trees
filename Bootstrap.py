import math
import numpy as np
from problem2 import DT 
#-------------------------------------------------------------------------
'''
    Problem 3: Bagging: Boostrap Aggregation of decision trees (on continous attributes)
'''


#-----------------------------------------------
class Bag(DT):
    '''
        Bagging ensemble of Decision Tree (with contineous attributes)
    '''

  
    #--------------------------
    @staticmethod
    def bootstrap(X,Y):
        '''
            Creating a boostrap sample of the dataset. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                X: the bootstrap sample of the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the bootstrap sample of the class labels, a numpy array of length n. Each element can be int/float/string.
        '''

        n = X.shape[1]
        
        indices = np.random.choice(np.arange(n),n,replace=True)
        
        X = X.T[indices].T
        
        Y = Y[indices]

        return X, Y 
    
    
    
    #--------------------------
    def train(self, X, Y, n_tree=11):
        '''
            Given a training set, training a bagging ensemble of decision trees. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                n_tree: the number of trees in the ensemble
            Output:
                T: a list of the root of each tree, a list of length n_tree.
        '''

        T =[]

        for i in range(n_tree):
            X1,Y1 = Bag.bootstrap(X,Y)
            T.append(DT.train(self, X,Y))

        return T 
    
    
    #--------------------------
    @staticmethod
    def inference(T,x):
        '''
            Given a bagging ensemble of decision trees and one data instance, infering the label of the instance. 
            Input:
                T: a list of decision trees.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''

        Y =[]
        
        for t in T:
            inference = DT.inference(t,x)
            Y.append(inference)
        y = max(set(Y),key=Y.count)

        return y
    
    
    
    #--------------------------
    @staticmethod
    def predict(T,X):
        '''
            Given a decision tree and a dataset, predicting the labels on the dataset. 
            Input:
                T: a list of decision trees.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''

        Y = []
        
        for i in X.T:
            Y.append(Bag.inference(T,i))

        return np.array(Y)
    
    
    #--------------------------
  
