#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 03:56:43 2022
Collaborative Matrix Factorization

[1] X. Zheng, H. Ding, H. Mamitsuka, and S. Zhu, "Collaborative matrix factorization with multiple similarities for predicting drug-target interaction", KDD, 2013.

This script was created based on PyDTI developed by Liu et al. and NRLMFb developed by Tomohiro.B
- https://github.com/stephenliu0423/PyDTI
- https://github.com/akiyamalab/NRLMFb

@author: docker
"""
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

class CMF:
    def __init__(self,num_factors=50,lambda_l=0.01,lambda_d=0.01,lambda_t=0.01,max_iter=5000,tolx=1e-4,verbose=False,half_mask=False):
        self.num_factors = num_factors
        self.lambda_l = lambda_l
        self.lambda_d = lambda_d
        self.lambda_t = lambda_t
        self.max_iter = max_iter
        self.tolx = tolx
        self.verbose = verbose
        self.half_mask = half_mask

    def fix_model(self, W, intMat, drugMat, targetMat, seed=None):
        self.num_drugs, self.num_targets = intMat.shape
        self.drugMat, self.targetMat = drugMat, targetMat
        x, y = np.where(W > 0)
        self.train_drugs = set(x.tolist())
        self.train_targets = set(y.tolist())
        if seed is None:
            self.U = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_targets, self.num_factors))
        else:
            prng = np.random.RandomState(seed)
            self.U = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self.num_targets, self.num_factors))
        self.ones = np.identity(self.num_factors)
        last_loss = self.compute_loss(W, intMat, drugMat, targetMat)
        
        if self.half_mask:
            rev_half = np.where(W == 1,0,0.5) # reverse the mask matrix
            self.WR = intMat*W + rev_half # mask
            #print(np.count_nonzero(self.R==0.5))
        else:
            self.WR = intMat*W # mask
        
        full_W = np.ones(W.shape)
        pbar = tqdm(range(self.max_iter))
        for t in pbar:
            #self.U = self.als_update(self.U, self.V, W, self.WR, drugMat, self.lambda_l, self.lambda_d)
            #self.V = self.als_update(self.V, self.U, W.T, self.WR.T, targetMat, self.lambda_l, self.lambda_t)
            #curr_loss = self.compute_loss(W, intMat, drugMat, targetMat)
            self.U = self.als_update(self.U, self.V, full_W, self.WR, drugMat, self.lambda_l, self.lambda_d)
            self.V = self.als_update(self.V, self.U, full_W.T, self.WR.T, targetMat, self.lambda_l, self.lambda_t)
            curr_loss = self.compute_loss(full_W, intMat, drugMat, targetMat)
            delta_loss = (curr_loss-last_loss)/last_loss
            
            pbar.set_postfix(DELTA=delta_loss,LOSS=curr_loss)
            # print "Epoach:%s, Curr_loss:%s, Delta_loss:%s" % (t+1, curr_loss, delta_loss)
            if abs(delta_loss) < self.tolx:
                if self.verbose:
                    print("--- Early Stopping ---")
                    print("iter :",t,"delta :",delta_loss)
                else:
                    pass
                break
            last_loss = curr_loss

    def als_update(self, U, V, W, R, S, lambda_l, lambda_d):
        X = R.dot(V) + 2*lambda_d*S.dot(U)
        Y = 2*lambda_d*np.dot(U.T, U)
        Z = lambda_d*(np.diag(S)-np.sum(np.square(U), axis=1))
        U0 = np.zeros(U.shape)
        D = np.dot(V.T, V)
        m, n = W.shape
        for i in range(m):
            # A = np.dot(V.T, np.diag(W[i, :]))
            # B = A.dot(V) + Y + (lambda_l+Z[i])*self.ones
            ii = np.where(W[i, :] > 0)[0]
            if ii.size == 0:
                B = Y + (lambda_l+Z[i])*self.ones
            elif ii.size == n:
                B = D + Y + (lambda_l+Z[i])*self.ones
            else:
                A = np.dot(V[ii, :].T, V[ii, :])
                B = A + Y + (lambda_l+Z[i])*self.ones
            U0[i, :] = X[i, :].dot(np.linalg.inv(B))
        return U0

    def compute_loss(self, W, intMat, drugMat, targetMat):
        loss = np.linalg.norm(W * (intMat - np.dot(self.U, self.V.T)), "fro")**(2)
        loss += self.lambda_l*(np.linalg.norm(self.U, "fro")**(2)+np.linalg.norm(self.V, "fro")**(2))
        loss += self.lambda_d*np.linalg.norm(drugMat-self.U.dot(self.U.T), "fro")**(2)+self.lambda_t*np.linalg.norm(targetMat-self.V.dot(self.V.T), "fro")**(2)
        return 0.5*loss

    def evaluation(self, test_data, test_label):
        ii, jj = test_data[:, 0], test_data[:, 1]
        self.scores = np.sum(self.U[ii, :]*self.V[jj, :], axis=1)
        prec, rec, thr = precision_recall_curve(test_label, self.scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, self.scores)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val
    
    def summarize(self):          
        # plot original matrix and completed one
        sns.heatmap(self.WR)
        plt.title("original R")
        plt.show()
        
        sns.heatmap(self.U.dot(self.V.T))
        plt.title("R' ~ W * H")
        plt.show()
