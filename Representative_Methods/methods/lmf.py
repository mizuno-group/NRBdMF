#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 09:01:44 2022

This script was created based on logistic-mf developed by Chris Johnson
https://github.com/MrChrisJohnson/logistic-mf

AdaFrad update

@author: docker
"""
import copy
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc

class LMF():
    def __init__(self,num_factors=50,max_iter=5000,reg_param=0.6,gamma=1.0,tolx=1e-4,verbose=False,half_mask=False):
        self.num_factors = int(num_factors)
        self.max_iter = int(max_iter)
        self.reg_param = float(reg_param)
        self.gamma = float(gamma)
        self.tolx = tolx
        self.verbose = verbose
        self.half_mask = half_mask
    
    def set_data(self,Y,W=None):
        """
        Y ~ X1WHtX2t
        for train_model() single run
        """
        if W is None:
            W = np.ones(Y.shape)
        self.intMat = W*Y
        self.num_drugs = self.intMat.shape[0]
        self.num_targets = self.intMat.shape[1]
        self.U = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_drugs, self.num_factors))
        self.V = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_targets, self.num_factors))
    
    def train_model(self):
        self.ones = np.ones((self.num_drugs, self.num_targets))
        self.drug_biases = np.random.normal(size=(self.num_drugs, 1))
        self.target_biases = np.random.normal(size=(self.num_targets, 1))

        drug_vec_deriv_sum = np.zeros((self.num_drugs, self.num_factors))
        target_vec_deriv_sum = np.zeros((self.num_targets, self.num_factors))
        drug_bias_deriv_sum = np.zeros((self.num_drugs, 1))
        target_bias_deriv_sum = np.zeros((self.num_targets, 1))
        pbar = tqdm(range(self.max_iter))
        for i in pbar:
            #t0 = time.time()
            # Fix targets and solve for drugs
            # take step towards gradient of deriv of log likelihood
            # we take a step in positive direction because we are maximizing LL
            drug_vec_deriv, drug_bias_deriv = self.deriv(True)
            drug_vec_deriv_sum += np.square(drug_vec_deriv)
            drug_bias_deriv_sum += np.square(drug_bias_deriv)
            vec_step_size = self.gamma / np.sqrt(drug_vec_deriv_sum)
            bias_step_size = self.gamma / np.sqrt(drug_bias_deriv_sum)
            self.U += vec_step_size * drug_vec_deriv
            self.drug_biases += bias_step_size * drug_bias_deriv

            # Fix drugs and solve for targets
            # take step towards gradient of deriv of log likelihood
            # we take a step in positive direction because we are maximizing LL
            target_vec_deriv, target_bias_deriv = self.deriv(False)
            target_vec_deriv_sum += np.square(target_vec_deriv)
            target_bias_deriv_sum += np.square(target_bias_deriv)
            vec_step_size = self.gamma / np.sqrt(target_vec_deriv_sum)
            bias_step_size = self.gamma / np.sqrt(target_bias_deriv_sum)
            self.V += vec_step_size * target_vec_deriv
            self.target_biases += bias_step_size * target_bias_deriv
            #t1 = time.time()
            #print('iteration %i finished in %f seconds' % (i + 1, t1 - t0))
    
    def fix_model(self, W, intMat, drugMat=None, targetMat=None, seed=None):
        """without set_data module"""
        self.num_drugs, self.num_targets = intMat.shape
        
        if self.half_mask:
            rev_half = np.where(W == 1,0,0.5) # reverse the mask matrix
            self.intMat = intMat*W + rev_half # mask
            #print(np.count_nonzero(self.R==0.5))
        else:
            self.intMat = intMat*W # mask

        x, y = np.where(self.intMat > 0)
        
        if seed is None:
            self.U = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_targets, self.num_factors))
        else:
            prng = np.random.RandomState(seed)
            self.U = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self.num_targets, self.num_factors))
        
        self.ones = np.ones((self.num_drugs, self.num_targets))
        self.drug_biases = np.random.normal(size=(self.num_drugs, 1))
        self.target_biases = np.random.normal(size=(self.num_targets, 1))

        drug_vec_deriv_sum = np.zeros((self.num_drugs, self.num_factors))
        target_vec_deriv_sum = np.zeros((self.num_targets, self.num_factors))
        drug_bias_deriv_sum = np.zeros((self.num_drugs, 1))
        target_bias_deriv_sum = np.zeros((self.num_targets, 1))
        
        sqrteps = np.sqrt(np.spacing(1))
        pbar = tqdm(range(self.max_iter))
        for i in pbar:
            last_U = copy.deepcopy(self.U)
            last_V = copy.deepcopy(self.V)
            #t0 = time.time()
            # Fix targets and solve for drugs
            # take step towards gradient of deriv of log likelihood
            # we take a step in positive direction because we are maximizing LL
            drug_vec_deriv, drug_bias_deriv = self.deriv(True)
            drug_vec_deriv_sum += np.square(drug_vec_deriv)
            drug_bias_deriv_sum += np.square(drug_bias_deriv)
            vec_step_size = self.gamma / np.sqrt(drug_vec_deriv_sum)
            bias_step_size = self.gamma / np.sqrt(drug_bias_deriv_sum)
            self.U += vec_step_size * drug_vec_deriv
            self.drug_biases += bias_step_size * drug_bias_deriv

            # Fix drugs and solve for targets
            # take step towards gradient of deriv of log likelihood
            # we take a step in positive direction because we are maximizing LL
            target_vec_deriv, target_bias_deriv = self.deriv(False)
            target_vec_deriv_sum += np.square(target_vec_deriv)
            target_bias_deriv_sum += np.square(target_bias_deriv)
            vec_step_size = self.gamma / np.sqrt(target_vec_deriv_sum)
            bias_step_size = self.gamma / np.sqrt(target_bias_deriv_sum)
            self.V += vec_step_size * target_vec_deriv
            self.target_biases += bias_step_size * target_bias_deriv
            #t1 = time.time()
            #print('iteration %i finished in %f seconds' % (i + 1, t1 - t0))
            
            # calc delta
            # epsilon based on machine precision
            du = np.amax(abs(self.U-last_U) / (sqrteps+np.amax(abs(last_U))))
            dv = np.amax(abs(self.V-last_V) / (sqrteps+np.amax(abs(last_V))))
            delta = max(du,dv)
            
            pbar.set_postfix(DELTA=delta)
            
            if delta <= self.tolx:
                if self.verbose:
                    print("--- meeted the tolx ---")
                    print(i,"/",self.max_iter,"iterations")
                break
        
    def evaluation(self,test_data, test_label):
        ii, jj = test_data[:,0], test_data[:,1]
        self.scores = np.sum(self.U[ii,:]*self.V[jj,:], axis=1)
        prec, rec, thr = precision_recall_curve(test_label, self.scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, self.scores)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val
        
    def evaluation_tmp(self, test_data, test_label):
        dinx = np.array(list(self.train_drugs))
        DS = self.dsMat[:, dinx]
        tinx = np.array(list(self.train_targets))
        TS = self.tsMat[:, tinx]
        scores = []
        if self.K2 > 0:
            for d, t in test_data:
                if d in self.train_drugs:
                    if t in self.train_targets:
                        val = np.sum(self.U[d, :]*self.V[t, :])
                    else:
                        jj = np.argsort(TS[t, :])[::-1][:self.K2]
                        val = np.sum(self.U[d, :]*np.dot(TS[t, jj], self.V[tinx[jj], :]))/np.sum(TS[t, jj])
                else:
                    if t in self.train_targets:
                        ii = np.argsort(DS[d, :])[::-1][:self.K2]
                        val = np.sum(np.dot(DS[d, ii], self.U[dinx[ii], :])*self.V[t, :])/np.sum(DS[d, ii])
                    else:
                        ii = np.argsort(DS[d, :])[::-1][:self.K2]
                        jj = np.argsort(TS[t, :])[::-1][:self.K2]
                        v1 = DS[d, ii].dot(self.U[dinx[ii], :])/np.sum(DS[d, ii])
                        v2 = TS[t, jj].dot(self.V[tinx[jj], :])/np.sum(TS[t, jj])
                        val = np.sum(v1*v2)
                scores.append(np.exp(val)/(1+np.exp(val)))
        elif self.K2 == 0:
            for d, t in test_data:
                val = np.sum(self.U[d, :]*self.V[t, :])
                scores.append(np.exp(val)/(1+np.exp(val)))
        prec, rec, thr = precision_recall_curve(test_label, np.array(scores))
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, np.array(scores))
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val
        
    def summarize(self):
        sns.heatmap(self.intMat)
        plt.title("original Y")
        plt.show()
        sns.heatmap(self.U.dot(self.V.T))
        plt.title("Y* ~ W H")
        plt.show()

# functions
    def deriv(self, drug):
        if drug:
            vec_deriv = np.dot(self.intMat, self.V)
            bias_deriv = np.expand_dims(np.sum(self.intMat, axis=1), 1)

        else:
            vec_deriv = np.dot(self.intMat.T, self.U)
            bias_deriv = np.expand_dims(np.sum(self.intMat, axis=0), 1)
        A = np.dot(self.U, self.V.T)
        A += self.drug_biases
        A += self.target_biases.T
        A = np.exp(A) 
        A /= (A + self.ones) # exp(xuyiT + Bu + Bi) / (1 + exp(xuyiT + Bu + Bi))
        A = (self.intMat + self.ones) * A

        if drug:
            vec_deriv -= np.dot(A, self.V)
            bias_deriv -= np.expand_dims(np.sum(A, axis=1), 1)
            # L2 regularization
            vec_deriv -= self.reg_param * self.U
        else:
            vec_deriv -= np.dot(A.T, self.U)
            bias_deriv -= np.expand_dims(np.sum(A, axis=0), 1)
            # L2 regularization
            vec_deriv -= self.reg_param * self.V
        return (vec_deriv, bias_deriv)

    def log_likelihood(self):
        loglik = 0
        A = np.dot(self.U, self.V.T)
        A += self.drug_biases
        A += self.target_biases.T
        B = A * self.intMat
        loglik += np.sum(B)

        A = np.exp(A)
        A += self.ones

        A = np.log(A)
        A = (self.intMat + self.ones) * A
        loglik -= np.sum(A)

        # L2 regularization
        loglik -= 0.5 * self.reg_param * np.sum(np.square(self.U))
        loglik -= 0.5 * self.reg_param * np.sum(np.square(self.V))
        return loglik

    def print_vectors(self):
        drug_vecs_file = open('logmf-drug-vecs-%i' % self.num_factors, 'w')
        for i in range(self.num_drugs):
            vec = ' '.join(map(str, self.U[i]))
            line = '%i\t%s\n' % (i, vec)
            drug_vecs_file.write(line)
        drug_vecs_file.close()
        target_vecs_file = open('logmf-target-vecs-%i' % self.num_factors, 'w')
        for i in range(self.num_targets):
            vec = ' '.join(map(str, self.V[i]))
            line = '%i\t%s\n' % (i, vec)
            target_vecs_file.write(line)
        target_vecs_file.close()