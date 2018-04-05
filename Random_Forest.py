import math
import numpy as np
from problem3 import Bag
#-------------------------------------------------------------------------
'''
    Problem 4: Random Forest (on continous attributes)
'''


#-----------------------------------------------
class RF(Bag):
    '''
        Random Forest (with contineous attributes)
        Random Forest is a subclass of Bagging class in problem3. So we can reuse and overwrite the code in problem 3.
    '''
  
    #--------------------------
    def best_attribute(self, X,Y):
        '''
            Finding the best attribute to split the node. (Overwritting the best_attribute function in the parent class: DT).
            The attributes have continous values (int/float).
            Here only a random sample of m features are considered. m = floor(sqrt(p)).
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''

        m = int(np.floor(np.sqrt(X.shape[0])))
        
        indices = np.random.choice(np.arange(X.shape[0]),m)
        
        i,th = Bag.best_attribute(self,X[indices],Y)
        
        if th == -np.inf:
            return RF.best_attribute(self, X,Y)

        return i, th
