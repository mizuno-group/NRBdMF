#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 07:44:48 2022

Galeano, D., Li, S., Gerstein, M & Paccanaro, A. (2020). Predicting the frequencies to drug side effects. Nature Communications.

https://www.nature.com/articles/s41467-020-18305-y
matlab version : https://github.com/paccanarolab/Side-effect-Frequencies

@author: docker
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc

class DiegoNMF():
    def __init__(self,num_factors=50,alpha=0.05,tolx=1e-4,max_iter=5000,variance=0.01,verbose=False,half_mask=False):
        self.R = None
        self.num_factors=num_factors
        self.alpha=alpha
        self.tolx=tolx
        self.max_iter=max_iter
        self.variance=variance
        self.verbose=verbose
        self.half_mask=half_mask
    
    def set_data(self,data):
        """
        Matrix to be decomposed
        e.g. drugs in row and side-effects in column
        """
        self.R = np.array(data)
    
    def fix_model(self, W, intMat, drugMat=None, targetMat=None, seed=None):
        """perform matrix factorization"""
        self.num_drugs, self.num_targets = intMat.shape
        if self.half_mask:
            rev_half = np.where(W == 1,0,0.5) # reverse the mask matrix
            self.R = intMat*W + rev_half # mask
            #print(np.count_nonzero(self.R==0.5))
        else:
            self.R = intMat*W # mask
        # initialization
        #np.random.seed(123)
        if seed is None:
            W0 = np.random.random((self.num_drugs, self.num_factors))*np.sqrt(self.variance)
            H0 = np.random.random((self.num_factors, self.num_targets))*np.sqrt(self.variance)
        else:
            np.random.seed(seed)
            W0 = np.random.random((self.num_drugs, self.num_factors))*np.sqrt(self.variance)
            H0 = np.random.random((self.num_factors, self.num_targets))*np.sqrt(self.variance)
        # normalization
        H0 = (H0.T / np.sqrt(np.sum(H0*H0,axis=1))).T
        """
        test1 = np.array([[1,2,3],[4,5,6]])
        test2 = np.array([6,15])
        test3 = (test1.T / test2).T = [[0.16667,0.33333,0.5],[0.26667,0.33333,0.4]]
        """
        # epsilon based on machine precision
        sqrteps = np.sqrt(np.spacing(1))
        
        # filter for clinical trials values
        CT = self.R > 0
        # filter for unobserved associations
        UN = self.R == 0
        
        self.costs = []
        self.deltas = []
        
        pbar = tqdm(range(self.max_iter))
        last_log = self.calc_loss(W0,H0)
        for i in pbar:
            # update W
            numer = (CT*self.R).dot(H0.T) # CT*R is same to original R
            denom = (CT*(W0.dot(H0)) + self.alpha*UN*W0.dot(H0)).dot(H0.T) + np.spacing(numer) # matlab eps = numpy spacing
            # W = max(0,W0*(numer/denom))
            W = W0*(numer/denom)
            
            # update H
            numer = W.T.dot((CT*self.R))
            denom = W.T.dot(CT*(W.dot(H0)) + self.alpha*UN*W.dot(H0)) + np.spacing(numer)
            # H = max(0,H0*(numer/denom))
            H = H0*(numer/denom)
            
            # compute cost
            cost = 0.5*(np.linalg.norm(CT*(self.R-W.dot(H)),ord="fro")**2) + 0.5*self.alpha*(np.linalg.norm(UN*(self.R-W.dot(H)),ord="fro")**2)
            self.costs.append(cost)
            
            # get norm of difference and max change in factors
            dw = np.amax(abs(W-W0) / (sqrteps+np.amax(abs(W0))))
            dh = np.amax(abs(H-H0) / (sqrteps+np.amax(abs(H0))))
            delta = max(dw,dh)
            self.deltas.append(delta)
            
            curr_log = self.calc_loss(W,H)
            loss = (curr_log-last_log)/abs(last_log)
            last_log = curr_log
            
            pbar.set_postfix(DELTA=delta,LOSS=loss)
            
            if i > 1:
                if delta <= self.tolx:
                    if self.verbose:
                        print("---completed---")
                        print("iter :",i,"delta :",delta)
                    break
            """
            if i > 1:
                if loss <= self.tolx:
                    if self.verbose:
                        print("---completed---")
                        print("iter :",i,"delta :",delta)
                    break
            """
            #if self.verbose:
                #print("iter :",i,"delta :",delta)
            W0 = W
            H0 = (H.T / np.sqrt(np.sum(H*H,axis=1))).T
        # return
        self.W = W0
        self.H = H0
    
    def calc_loss(self,W,H):
        loss = 0
        tmp = self.R - np.dot(W,H)
        loss -= np.sum(tmp*tmp)
        return loss
    
    def evaluation(self,test_data, test_label):
        ii, jj = test_data[:,0], test_data[:,1]
        self.scores = np.sum(self.W[ii,:]*self.H.T[jj,:], axis=1) # note that H is transposed
        prec, rec, thr = precision_recall_curve(test_label, self.scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, self.scores)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val
    
    def summarize(self,ntop=200):
        # plot the delta and the loss distribution
        use_delta = self.deltas[ntop:]
        x = [i+ntop for i in range(len(use_delta))]
        plt.plot(x,use_delta)
        plt.xlabel("iterations")
        plt.ylabel("delta")
        plt.show()
        
        use_cost = self.costs[ntop:]
        plt.plot(x,use_cost)
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.show()
        
        # plot original matrix and completed one
        sns.heatmap(self.R)
        plt.title("original R")
        plt.show()
        
        sns.heatmap(self.W.dot(self.H))
        plt.title("R' ~ W * H")
        plt.show()
