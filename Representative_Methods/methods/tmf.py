#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 06:53:34 2022

Triple Matrix Factorization (TMF)

A Novel Triple Matrix Factorization Method for Detecting Drug-Side Effect Association Based on Kernel Target Alignment
Xiaoyi.G. et al.
https://www.hindawi.com/journals/bmri/2020/4675395/


@author: docker
"""
import pandas as pd
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc

class TMF():
    def __init__(self,lamda=1.,verbose=False,half_mask=False):
        self.lamda = lamda
        self.verbose = verbose
        self.half_mask = half_mask
    
    def set_data(self,X1,X2,Y):
        """Y ~ X1MX2t"""
        self.X1 = X1
        self.X2 = X2
        self.Y = Y
        print("")
        print("---TMF---")
        print("Y :",self.Y.shape,end=" ")
        print("X1 :",self.X1.shape,end=" ")
        print("X2 :",self.X2.shape)
    
    def decompose(self):
        """perform matrix factorization"""
        a = self.X1.T.dot(self.X1)
        b = self.lamda*np.linalg.pinv(self.X2.T.dot(self.X2))
        c = self.X1.T.dot(self.Y).dot(np.linalg.pinv(self.X2.T))
        # solve sylvester equation
        self.M = scipy.linalg.solve_sylvester(a,b,c)
        
        # reconstruct Y*
        self.Y_res = self.X1.dot(self.M).dot(self.X2.T)
    
    def fix_model(self, W, intMat, drugMat, targetMat, seed=None):
        """
        intMat : (n,m)
        drugMat : (n,k1)
        targetMat : (k2,m)
        """
        self.W = W
        self.num_drugs, self.num_targets = intMat.shape
        
        if self.half_mask:
            rev_half = np.where(self.W == 1,0,0.5) # reverse the mask matrix
            self.intMat = intMat*self.W + rev_half # mask
        else:
            self.intMat = intMat*self.W # mask
        
        self.X1 = drugMat
        self.X2 = targetMat
        self.num_drugs = self.X1.shape[1]
        self.num_targets = self.X2.shape[0]
        
        """perform matrix factorization"""
        a = self.X1.T.dot(self.X1)
        b = self.lamda*np.linalg.pinv(self.X2.T.dot(self.X2))
        c = self.X1.T.dot(self.intMat).dot(np.linalg.pinv(self.X2.T))
        # solve sylvester equation
        self.M = scipy.linalg.solve_sylvester(a,b,c) # (k1,k2)
        
        # reconstruct intMat (Y)
        self.Y_res = self.X1.dot(self.M).dot(self.X2.T)
    
    def evaluation(self,test_data, test_label):
        """
        memory limitation
        scores = []
        for i in ii:
            for j in jj:
                scores.append(self.Y_res[i][j])
        """
        rev_W = np.where(self.W == 1,np.nan,1) # reverse the mask matrix
        score_mat = self.Y_res*rev_W
        self.scores = score_mat[~np.isnan(score_mat)] # same to np.array(pd.DataFrame(score_mat.flatten()).dropna())
        prec, rec, thr = precision_recall_curve(test_label, self.scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, self.scores)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val
    
    def ex_evaluation(self,test_intMat,test_drugMat,test_targetMat,do_plot=False):
        # predict with external dataset
        self.Y_pred = test_drugMat.dot(self.M).dot(test_targetMat.T)
        # flatten
        test_label = test_intMat.flatten()
        self.scores = self.Y_pred.flatten()
        
        self.pred_res = pd.DataFrame({"scores":self.scores,"label":test_label}).sort_values("scores",ascending=False)
        
        prec, rec, thr = precision_recall_curve(test_label, self.scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, self.scores)
        auc_val = auc(fpr, tpr)
        if do_plot:
            sns.heatmap(test_intMat.T)
            plt.title("original test Y")
            plt.show()
            sns.heatmap(self.Y_pred.T)
            plt.title("Y' ~ X1 * M * X2t")
            plt.show()
        return aupr_val, auc_val
    
    def summarize(self):
        # plot original matrix and completed one
        sns.heatmap(self.intMat)
        plt.title("original Y")
        plt.show()
        
        sns.heatmap(self.Y_res)
        plt.title("Y' ~ X1 * M * X2t")
        plt.show()