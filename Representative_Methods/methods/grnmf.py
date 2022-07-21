#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 07:13:13 2022

Graph Regularized NMF:

[3] Cai, D., He, X., Han, J., & Huang, T. S. (2011). Graph regularized
	nonnegative matrix factorization for data representation. IEEE Transactions
	on Pattern Analysis and Machine Intelligence, 33(8), 1548-1560.

@author: docker
"""
import numpy as np
import numpy.linalg as LA
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc

class GRNMF():
    def __init__(self,num_factors=50,max_iter=5000,variance=0.01,sigma=0.3,tolx=1e-4,verbose=False,half_mask=False):
        self.intMat = None
        self.num_factors=num_factors
        self.max_iter=max_iter
        self.variance=variance
        self.sigma=sigma
        self.tolx = tolx
        self.verbose = verbose
        self.half_mask = half_mask
    
    def fix_model(self, W, intMat, drugMat=None, targetMat=None, seed=None):
        self.num_drugs, self.num_targets = intMat.shape
        if self.half_mask:
            rev_half = np.where(W == 1,0,0.5) # reverse the mask matrix
            self.X = intMat*W + rev_half # mask
        else:
            self.X = intMat*W # mask
        # initialization
        if seed is None:
            self.W = np.random.random((self.num_drugs, self.num_factors))*np.sqrt(self.variance)
            self.H = np.random.random((self.num_factors, self.num_targets))*np.sqrt(self.variance)
        else:
            np.random.seed(seed)
            self.W = np.random.random((self.num_drugs, self.num_factors))*np.sqrt(self.variance)
            self.H = np.random.random((self.num_factors, self.num_targets))*np.sqrt(self.variance)
            
        self.compute_factors(lmd=0, weight_type='heat-kernel')
    
    def compute_graph(self, weight_type='heat-kernel'):
        if weight_type == 'heat-kernel':
            samples = np.matrix(self.X.T)
            A= np.zeros((samples.shape[0], samples.shape[0]))

            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    A[i][j]= np.exp(-(LA.norm(samples[i] - samples[j] ))/self.sigma )
            return A
        
        elif weight_type == 'dot-weighting':
            samples = np.matrix(self.X.T)
            A= np.zeros((samples.shape[0], samples.shape[0]))

            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    A[i][j]= np.dot(samples[i],samples[j])
            return A
    
    def compute_factors(self, lmd=0, weight_type='heat-kernel'):
  
        A = self.compute_graph(weight_type)
  
        D = np.matrix(np.diag(np.asarray(A).sum(axis=0)))
  
        self.frob_error = np.zeros(self.max_iter)
  
        for i in range(self.max_iter):
            dw = self.update_w(lmd, A, D)
            dh = self.update_h(lmd, A, D)
            self.frob_error[i] = self.frobenius_norm()
            # get norm of difference and max change in factors
            delta = max(dw,dh)
            if delta <= self.tolx:
                if self.verbose:
                    print("--- meeted the tolx ---")
                    print(i,"/",self.max_iter,"iterations")
                break
    
    def update_h(self, lmd, A, D):
        last_H = self.H
        eps = 2**-8
        h_num = lmd*np.dot(A, self.H.T)+np.dot(self.X.T, self.W )
        h_den = lmd*np.dot(D, self.H.T)+np.dot(self.H.T, np.dot(self.W.T, self.W))

        self.H = np.multiply(self.H.T, (h_num+eps)/(h_den+eps))
        self.H = self.H.T
        self.H[self.H <= 0] = eps
        self.H[np.isnan(self.H)] = eps
        
        # epsilon based on machine precision
        sqrteps = np.sqrt(np.spacing(1))
        dh = np.amax(abs(self.H-last_H) / (sqrteps+np.amax(abs(last_H))))
        return dh

    def update_w(self, lmd, A, D):
        last_W = self.W
        XH = self.X.dot(self.H.T)
        WHtH = self.W.dot(self.H.dot(self.H.T)) + 2**-8
        self.W *= XH
        self.W /= WHtH
        
        # epsilon based on machine precision
        sqrteps = np.sqrt(np.spacing(1))
        dw = np.amax(abs(self.W-last_W) / (sqrteps+np.amax(abs(last_W))))
        return dw

    def frobenius_norm(self):
        """ Euclidean error between X and W*H """
        if hasattr(self,'H') and hasattr(self,'W'):
            error = LA.norm(self.X - np.dot(self.W, self.H))
        else:
            error = None

        return error
    
    def evaluation(self,test_data, test_label):
        self.W = np.array(self.W) # note that type(self.H) == numpy.matrix
        self.H = np.array(self.H)
        ii, jj = test_data[:,0], test_data[:,1]
        self.scores = np.sum(self.W[ii,:]*self.H.T[jj,:], axis=1) # note that H is transposed
        prec, rec, thr = precision_recall_curve(test_label, self.scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, self.scores)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val
    
    def summarize(self):          
        # plot original matrix and completed one
        sns.heatmap(self.X)
        plt.title("original R")
        plt.show()
        
        sns.heatmap(self.W.dot(self.H))
        plt.title("R' ~ W * H")
        plt.show()