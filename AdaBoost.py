import math
import numpy as np
from problem2 import DT,Node
#-------------------------------------------------------------------------
'''
    Problem 5: Boosting (on continous attributes). 
               We will implement AdaBoost algorithm in this problem.
'''

#-----------------------------------------------
class DS(DT):
    '''
        Decision Stump (with contineous attributes) for Boosting.
        Decision Stump is also called 1-level decision tree.
        Different from other decision trees, a decision stump can have at most one level of child nodes.
        In order to be used by boosting, here we assume that the data instances are weighted.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y, D):
        '''
            Computing the entropy of the weighted instances.
            Input:
                Y: a list of labels of the instances, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the entropy of the weighted samples, a float scalar
        '''
        P = np.array([np.sum(D[np.where(Y == i)]) for i in np.unique(Y)])
        
        newP = P.copy()
        
        newP[newP == 0] = 1e-5
        
        e = np.sum(-np.multiply(P,np.log2(newP)))

        return e 

    @staticmethod
    def rescaleD(D):
        '''
            rescaling D to make sure that all weights in D add up to 1
            Input:
            D: Sample Weights which may or may not add up to 1
        '''

        if (np.zeros_like(D)==D).all():
            return np.ones_like(D)/len(D)
        return (1/np.sum(D))*D
    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X,D):
        '''
            Computing the conditional entropy of y given x on weighted instances
            Input:
                Y: a list of values, a numpy array of int/float/string values.
                X: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                ce: the weighted conditional entropy of y given x, a float scalar
        '''
        
        Probs1 = np.array([np.sum(D[X==i]) for i in np.unique(X)])
        
        e = np.array([DS.entropy(Y[X==i],DS.rescaleD(D[X==i])) for i in np.unique(X)])
        
        ce = np.sum(np.multiply(Probs1,e))

        return ce 

    #--------------------------
    @staticmethod
    def information_gain(Y,X,D):
        '''
            Computing the information gain of y after spliting over attribute x
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                g: the weighted information gain of y after spliting over x, a float scalar
        '''

        e1 = DS.entropy(Y,D)
        
        ce1 = DS.conditional_entropy(Y,X,D)
        
        g = e1 - ce1

        return g

    #--------------------------
    @staticmethod
    def best_threshold(X,Y,D):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X. The data instances are weighted. 
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n

            Output:
                th: the best threhold, a float scalar. 
                g: the weighted information gain by using the best threhold, a float scalar. 
        '''

        ig = lambda X,Y,threshold,D: DS.information_gain(Y,X>=threshold,D)
        ths = DT.cutting_points(X,Y)
    
        if np.all(ths == -np.inf):
            return -float('inf'),-1
        gs = [ig(X,Y,i,D) for i in ths]
        g = max(gs)
        th = ths[np.argmax(gs)]

        return th,g 
     
    #--------------------------
    def best_attribute(self,X,Y,D):
        '''
            Find the best attribute to split the node. The attributes have continous values (int/float). The data instances are weighted.
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''

        ths_gains = np.apply_along_axis(DS.best_threshold,1,X,Y,D)
        
        i = np.argmax(ths_gains[:,1])
        
        th = ths_gains[:,0][i]

        return i, th
             
    #--------------------------
    @staticmethod
    def most_common(Y,D):
        '''
            Get the most-common label from the list Y. The instances are weighted.
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
                D: the weights of instances, a numpy float vector of length n
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''

        y = np.unique(Y)[np.argmax([np.sum(D[Y == i]) for i in np.unique(Y)])]

        return y
 

    #--------------------------
    def build_tree(self, X,Y,D):
        '''
            build decision stump by overwritting the build_tree function in DT class.
            Instead of building tree nodes recursively in DT, here we only build at most one level of children nodes.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Return:
                t: the root node of the decision stump. 
        '''

        t = Node(X = X, Y = Y)
        t.p = DS.most_common(t.Y,D)
    
        # if Condition 1 or 2 holds, stop splitting
        if DS.stop1(t.Y) or DS.stop2(t.X):
            t.isleaf = True
            return t
        # find the best attribute to split
        
        t.i,t.th = DS.best_attribute(self,t.X,t.Y,D)
        
        t.C1 = Node(X=X.T[np.where(X[t.i]<t.th)].T,Y=Y[X[t.i]<t.th])
        
        t.C2 = Node(X=X.T[np.where(X[t.i]>=t.th)].T,Y=Y[X[t.i]>=t.th])
        
        t.C1.p = DS.most_common(t.C1.Y,D[X[t.i]<t.th])
        
        t.C2.p = DS.most_common(t.C2.Y,D[X[t.i]>=t.th])
        
        t.C1.isleaf = True
        
        t.C2.isleaf = True

        return t
    
 

#-----------------------------------------------
class AB(DS):
    '''
        AdaBoost algorithm (with contineous attributes).
    '''

    #--------------------------
    @staticmethod
    def weighted_error_rate(Y,Y_,D):
        '''
            Computing the weighted error rate of a decision on a dataset. 
            Input:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the weighted error rate of the decision stump
        '''

        e = np.sum(D[Y != Y_])

        return e

    #--------------------------
    @staticmethod
    def compute_alpha(e):
        '''
            Computing the weight a decision stump based upon weighted error rate.
            Input:
                e: the weighted error rate of a decision stump
            Output:
                a: (alpha) the weight of the decision stump, a float scalar.
        '''

        if(e == 0):
            return np.log(np.finfo(float).max)
        
        elif(e == 1):
            return -np.log(np.finfo(float).max)
        
        a = 0.5*np.log((1-e)/e)

        return a

    #--------------------------
    @staticmethod
    def update_D(D,a,Y,Y_):
        '''
            updating the weight the data instances 
            Input:
                D: the current weights of instances, a numpy float vector of length n
                a: (alpha) the weight of the decision stump, a float scalar.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels by the decision stump, a numpy array of length n. Each element can be int/float/string.
            Output:
                D: the new weights of instances, a numpy float vector of length n
        '''

        D = D.copy()
        
        for i,x in enumerate(D):
            if Y[i] == Y_[i]:
                D[i] = x*np.exp(-a)
            else:
                D[i] = x*np.exp(a)
        
        D = AB.rescaleD(D)

        return D

    #--------------------------
    @staticmethod
    def step(X,Y,D):
        '''
            Computing one step of Boosting.  
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the current weights of instances, a numpy float vector of length n
            Output:
                t:  the root node of a decision stump trained in this step
                a: (alpha) the weight of the decision stump, a float scalar.
                D: the new weights of instances, a numpy float vector of length n
        '''

        t = AB.build_tree(AB(),X,Y,D)
        
        Y_ = DS.predict(t,X)
        
        e = AB.weighted_error_rate(Y,Y_,D)
        
        a = AB.compute_alpha(e)
        
        D = AB.update_D(D,a,Y,Y_)

        return t,a,D

    
    #--------------------------
    @staticmethod
    def inference(x,T,A):
        '''
            Given a bagging ensemble of decision trees and one data instance, infer the label of the instance. 
            Input:
                x: the attribute vector of a data instance, a numpy vectr of shape p.
                   Each attribute value can be int/float
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                y: the class label, a scalar of int/float/string.
        '''

        Y = np.array([DS.inference(t,x) for t in T])
        
        y = np.unique(Y)[np.argmax([np.sum(A[Y == i]) for i in np.unique(Y)])]

        return y
 

    #--------------------------
    @staticmethod
    def predict(X,T,A):
        '''
            Given an AdaBoost and a dataset, predict the labels on the dataset. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
        '''

        Y = np.array([AB.inference(x,T,A) for x in X.T])

        return Y 
 

    #--------------------------
    @staticmethod
    def train(X,Y,n_tree=10):
        '''
            train adaboost.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                n_tree: the number of trees in the ensemble, an integer scalar
            Output:
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
        '''
        # initialize weight as 1/n
        T = []
        A = []
        n = X.shape[1]
        D = np.ones(n)/n
        


        # iteratively build decision stumps
        for i in range(n_tree):
            t,a,D = AB.step(X,Y,D)
            T.append(t)
            A.append(a)
        
        A = np.array(A)

        return T, A
