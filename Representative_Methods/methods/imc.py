#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 04:40:31 2022

Inductive Matrix Completion

@author: docker
"""
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc

class IMC:
    def __init__(self,num_factors=50,reg_param1=1.,reg_param2=1.,variance=0.01,max_iter=5000,tolx=1e-4,verbose=False,half_mask=False):
        self.num_factors = num_factors
        self.reg_param1 = reg_param1
        self.reg_param2 = reg_param2
        self.variance = variance
        self.max_iter = max_iter
        self.tolx = tolx
        self.verbose = verbose
        self.half_mask = half_mask
    
    def set_data(self,A,X,Y):
        self.A = A
        self.X = X
        self.Y = Y
    
    def fix_model(self, W, intMat, drugMat, targetMat, seed=None, doplot=False):
        self.mask_W = W
        
        if self.half_mask:
            rev_half = np.where(self.mask_W == 1,0,0.5) # reverse the mask matrix
            self.A = intMat*self.mask_W + rev_half # mask
            #print(np.count_nonzero(self.R==0.5))
        else:
            self.A = intMat*self.mask_W # mask
        
        self.X = drugMat
        self.Y = targetMat
        # initialize W and H as rabdin dense matrix maintaining the non-negativity constraints Wik>=0, Hjk>=0
        if seed is None:
            self.W = np.random.random((self.X.shape[1], self.num_factors))*self.variance
            H = np.random.random((self.Y.shape[1], self.num_factors))*self.variance
            # normalzie
            self.H = (H.T / np.sqrt(np.sum(H*H,axis=1))).T
        else:
            np.random.seed(seed)
            self.W = np.random.random((self.X.shape[1], self.num_factors))*self.variance
            H = np.random.random((self.Y.shape[1], self.num_factors))*self.variance
            # normalzie
            self.H = (H.T / np.sqrt(np.sum(H*H,axis=1))).T

        loss = []
        self.deltas = []
        sqrteps = np.sqrt(np.spacing(1))
        for i in range(self.max_iter):
            last_H = np.array(copy.deepcopy(self.H))
            last_W = copy.deepcopy(self.W)
            H = update_h(self.A,self.X,self.Y,self.W,self.H,self.reg_param2)
            # normalzie
            self.H = (H.T / np.sqrt(np.sum(H*H,axis=1))).T
            self.W = update_w(self.A,self.X,self.Y,self.W,self.H,self.reg_param1)
            res = self.approximate()
            mse = calc_loss(self.A, res)
            loss.append(mse)
            
            # calc delta
            # epsilon based on machine precision
            dh = np.amax(abs(np.array(self.H)-last_H) / (sqrteps+np.amax(abs(last_H))))
            dw = np.amax(abs(self.W-last_W) / (sqrteps+np.amax(abs(last_W))))            
            delta = max(dw,dh)
            #print(dw,dh)
            self.deltas.append(delta)
            if delta <= self.tolx:
                if self.verbose:
                    print("--- meeted the tolx ---")
                    print(i,"/",self.max_iter,"iterations")
                break
        #print("final delta :",delta)
        if doplot:
            # plot the loss change
            plt.plot(loss)
            plt.xlabel("iterations")
            plt.ylabel("MSE")
            plt.show()
            # plot the loss change
            plt.plot(self.deltas)
            plt.xlabel("iterations")
            plt.ylabel("delta")
            plt.show()
        else:
            pass
        self.Y_res = ((self.X.dot(self.W)).dot(self.H.T)).dot(self.Y.T)
        
    def approximate(self):
        return ((self.X.dot(self.W)).dot(self.H.T)).dot(self.Y.T)
    
    def summarize(self):          
        # plot original matrix and completed one
        sns.heatmap(self.A)
        plt.title("original R")
        plt.show()
        
        sns.heatmap(self.Y_res)
        plt.title("R' ~ X1 * W * Ht * X2t")
        plt.show()
    
    def evaluation(self,test_data, test_label):
        """
        memory limitation
        scores = []
        for i in ii:
            for j in jj:
                scores.append(self.Y_res[i][j])
        """
        rev_W = np.where(self.mask_W == 1,np.nan,1) # reverse the mask matrix
        score_mat = self.Y_res*rev_W
        self.scores = score_mat[~np.isnan(score_mat)] # same to np.array(pd.DataFrame(score_mat.flatten()).dropna())
        prec, rec, thr = precision_recall_curve(test_label, self.scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, self.scores)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val
    
    def ex_evaluation(self,test_intMat,test_drugMat,test_targetMat):
        # predict with external dataset
        self.Y_pred = ((test_drugMat.dot(self.W)).dot(self.H.T)).dot(test_targetMat.T)
        # flatten
        test_label = test_intMat.flatten()
        self.scores = self.Y_pred.flatten()
        
        self.pred_res = pd.DataFrame({"scores":self.scores,"label":test_label}).sort_values("scores",ascending=False)
        
        prec, rec, thr = precision_recall_curve(test_label, self.scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, self.scores)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val
        
    
def update_h(A,X,Y,W,H,reg_param2):
    """
    fix W and update H
    H_jk <-- H_jk * (YAXW_jk / (YYHWXXW + reg_param2*H)_jk )
    """
    X2 = X.T.dot(X)
    Y2 = Y.T.dot(Y)
    HW = H.dot(W.T)
    # enumerator
    enum = ((Y.T.dot(A.T)).dot(X)).dot(W)
    # denominator
    YYHWXXW = ((Y2.dot(HW)).dot(X2)).dot(W)
    denom = YYHWXXW + reg_param2*H + np.spacing(enum)
    # update
    change = enum/denom
    H_new = H*change
    return H_new

def update_w(A,X,Y,W,H,reg_param1):
    """
    fix H and update W
    W_ik <-- W_ik * (XAYH_ik / (XXWHTTH + reg_param1*W)_ik )
    """
    X2 = X.T.dot(X)
    Y2 = Y.T.dot(Y)
    WH = W.dot(H.T)
    # enumerator
    enum = ((X.T.dot(A)).dot(Y)).dot(H)
    # denominator
    XXWHYYH = ((X2.dot(WH)).dot(Y2)).dot(H)
    denom = XXWHYYH + reg_param1*W + np.spacing(enum)
    # update
    change = enum/denom
    W_new = W*change
    return W_new

def calc_loss(A,res):
    """
    MSE : mean squared error
    """
    A_f = A.flatten(order='F')
    res_f = res.flatten(order='F')
    
    loss = mean_squared_error(A_f, res_f)
    return loss