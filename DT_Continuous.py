import math
import numpy as np
from problem1 import Tree
#-------------------------------------------------------------------------
'''
    Problem 2: Decision Tree (with continous attributes)
    In this problem, we will implement the decision tree method for classification problems.
'''

#--------------------------
class Node:
    '''
        Decision Tree Node (with continous attributes)
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar 
            th: the threshold on the attribute, a float scalar.
            C1: the child node for values smaller than threshold
            C2: the child node for values larger than threshold
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X=None,Y=None, i=None,th=None,C1=None, C2=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.th = th 
        self.C1= C1
        self.C2= C2
        self.isleaf = isleaf
        self.p = p


#-----------------------------------------------
class DT(Tree):
    '''
        Decision Tree (with contineous attributes)
        DT is a subclass of Tree class in problem1. So we can reuse and overwrite the code in problem 1.
    '''

    #--------------------------
    @staticmethod
    def cutting_points(X,Y):
        '''
            Finding all possible cutting points in the continous attribute of X. 
            (1) sort unique attribute values in X, like, x1, x2, ..., xn
            (2) consider splitting points of form (xi + x(i+1))/2 
            (3) only consider splitting between instances of different classes
            (4) if there is no candidate cutting point above, use -inf as a default cutting point.
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                cp: the list of  potential cutting points, a float numpy vector. 
        '''

        z = zip(X,Y)
        z.sort(key = lambda t: t[0])
        cp =[]
        
        for i in range(len(z)-1):
            for j in range(i+1,len(z)):
                if z[i][1] == z[j][1]:
                    break
                elif z[i][0] == z[j][0]:
                    continue
                else:
                    cp.append((z[i][0]+z[j][0])/2)
                    break
                
        if not len(cp):
            return -np.inf

        return np.array(cp)
    
    #--------------------------
    @staticmethod
    def best_threshold(X,Y):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X. 
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                th: the best threhold, a float scalar. 
                g: the information gain by using the best threhold, a float scalar. 
            Hint: you can reuse your code in problem 1.
        '''

        gains = []
        thresholds = DT.cutting_points(X,Y)
        if np.all(thresholds == -np.inf):
            return - float('inf'),-1
        for i in thresholds:
            gains.append(Tree.information_gain(Y,X>=i))
            
        g = max(gains)
        th=thresholds[np.argmax(gains)]

        return th,g 
    
    
    #--------------------------
    def best_attribute(self,X,Y):
        '''
            Finding the best attribute to split the node. The attributes have continous values (int/float).
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''

        thresholds_gains = np.apply_along_axis(DT.best_threshold,1,X,Y)
        i = np.argmax(thresholds_gains[:,1])
        th = thresholds_gains[:,0][i]

        return i, th
    
        
    #--------------------------
    @staticmethod
    def split(X,Y,i,th):
        '''
            Splitting the node based upon the i-th attribute and its threshold.
            (1) split the matrix X based upon the values in i-th attribute and threshold
            (2) split the labels Y 
            (3) build children nodes by assigning a submatrix of X and Y to each node
    
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C1: the child node for values smaller than threshold
                C2: the child node for values larger than (or equal to) threshold
        '''

        C1 = Node(X=X.T[X[i]<th].T,Y=Y[X[i]<th])
        C2 = Node(X=X.T[X[i]>=th].T,Y=Y[X[i]>=th])

        return C1, C2
    
    
    
    #--------------------------
    def build_tree(self, t):
        '''
            Recursively building tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape n by p.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C1: the child node for values smaller than threshold
                t.C2: the child node for values larger than (or equal to) threshold
        '''

    
        # if Condition 1 or 2 holds, stop recursion 
        t.p = DT.most_common(t.Y)
        
        
        # find the best attribute to split
        if Tree.stop1(t.Y) or Tree.stop2(t.X):
            t.isleaf = True
            return 
        
        t.i,t.th = self.best_attribute(t.X,t.Y)
        
        t.C1, t.C2 = DT.split(t.X,t.Y,t.i,t.th)
       
        
        # recursively build subtree on each child node
        self.build_tree(t.C1)
        self.build_tree(t.C2)
        return

    
    #--------------------------
    @staticmethod
    def inference(t,x):
        '''
            Given a decision tree and one data instance, infering the label of the instance recursively. 
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''

        while (not t.isleaf):
            if x[t.i]>=t.th:
                t = t.C2
            else:
                t = t.C1
                
        y = t.p

        return y
    
    
    #--------------------------
    @staticmethod
    def predict(t,X):
        '''
            Given a decision tree and a dataset, predicting the labels on the dataset. 
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''

        Y =[]
        
        for i in X.T:
            Y.append(DT.inference(t,i))

        return np.array(Y)
    
    
    
    #--------------------------
    def train(self, X, Y):
        '''
            Given a training set, training a decision tree. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''

        t = Node(X=X, Y=Y)
        DT.build_tree(self,t)

        return t


    #--------------------------
