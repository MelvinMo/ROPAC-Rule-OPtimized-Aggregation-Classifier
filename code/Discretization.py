from __future__ import division
import pandas as pd
import numpy as np
import sys


class Discretization:
        
    @staticmethod
    def basedOnEntropy(x : np.ndarray):
        labelSet = set(x[:,1]);        
        labelMap = {}
        t = 0;
        for element in labelSet:            
            labelMap[element] = t;
            t+=1;
        for i in range(0, len(x)):
            x[i, 1] = labelMap[x[i, 1]];
        
        sortedX = x[x[:, 0].argsort()];        
        featureValues = sortedX[:, 0];
        featureValues = np.unique(featureValues);
        if featureValues.shape[0] < 2:
            print("one feature had unique value -> shape : {0}".format(featureValues.shape[0]));
            return np.zeros(x.shape[0], dtype=int);
        
        condidateCutPoints = np.zeros(len(featureValues)-1);
        
        for i in range(0, len(featureValues)-1):            
            condidateCutPoints[i] = (featureValues[i] + featureValues[i+1])/2;
        
        selectedCutPointIndex = -1;
        minEntropy = sys.float_info.max;
        for i in range(0, len(condidateCutPoints)):
            ent = Discretization.__entropy(sortedX, condidateCutPoints[i], len(labelSet));            
            if ent < minEntropy :
                minEntropy = ent;
                selectedCutPointIndex = i;
        #print("cutPoint : {0}".format(condidateCutPoints[selectedCutPointIndex]));
        return Discretization.__generateNewData(x[:,0], condidateCutPoints[selectedCutPointIndex]);
        
    @staticmethod
    def __generateNewData(data : np.ndarray, cutpoint):
        newData = np.zeros(len(data), dtype=int);
        for i in range(0, len(data)):
            if(data[i] <= cutpoint):
                newData[i] = 0;
            else:
                newData[i] = 1;
        return newData;
        
    @staticmethod
    def __entropy(data : np.ndarray, cutpoint, numOfClasses):
        entropy = 0;
        leftBucketInstanceNumber = 0;
        rightBucketInstanceNumber = 0;
        allInstanceNumber = len(data);
        leftBucket = np.zeros(numOfClasses);
        rightBucket = np.zeros(numOfClasses);
        for i in range(0, len(data)):
            if data[i, 0] <= cutpoint :
                leftBucket[data[i, 1]] += 1;
                leftBucketInstanceNumber += 1;
            else:
                rightBucket[data[i, 1]] += 1;
                rightBucketInstanceNumber += 1;

        
        leftBucketClassProbability = leftBucket / leftBucketInstanceNumber;
        rightBucketClassProbability = rightBucket / rightBucketInstanceNumber;
        leftBucketEntropy = np.sum(leftBucketClassProbability * np.log2(leftBucketClassProbability, out=np.zeros_like(leftBucketClassProbability), where=(leftBucketClassProbability!=0)));
        rightBucketEntropy = np.sum(rightBucketClassProbability * np.log2(rightBucketClassProbability, out=np.zeros_like(rightBucketClassProbability), where=(rightBucketClassProbability!=0)));

        entropy = (leftBucketInstanceNumber / allInstanceNumber) * leftBucketEntropy + (rightBucketInstanceNumber / allInstanceNumber) * rightBucketEntropy;
        entropy = -1 * entropy;        
        return entropy;
    




'''
dataset = np.array([[85, 80, 83, 70, 68, 65, 64, 72, 69, 75, 75, 72, 81, 71],
           [85, 90, 78, 96, 80, 70, 65, 95, 70, 80, 70, 90, 75, 80],
           ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']], dtype = object);
dataset = dataset.T;
print("dataset before discretization : \n{0}".format(dataset));
dataset[:, 0] = Discretization.basedOnEntropy(dataset[:, [0,2]]);
dataset[:, 1] = Discretization.basedOnEntropy(dataset[:, [1,2]]);
print("dataset after discretization : \n{0}".format(dataset));
'''