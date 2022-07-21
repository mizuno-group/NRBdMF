#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:17:20 2022

Neigborhood Regularized Bidirectional Matrix Factorization (NRBdMF)

(arg⁡min)┬(U,V)⁡J=   1/2 ‖R ° (Y -UV^T)‖_F^2  +  1/2 tr[U^T (λ_d I+ αL_d )U]  +  1/2 tr[V^T (λ_t I+ βL_t )V]

This algorithm was inspired by NRLMF.
- Yong Liu et al., PLOS Comput. Biol., 2016
- https://doi.org/10.1371/journal.pcbi.1004760

This script was created based on PyDTI developed by Liu et al. and NRLMFb developed by Tomohiro.B
- https://github.com/stephenliu0423/PyDTI
- https://github.com/akiyamalab/NRLMFb

@author: docker
"""
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score

monitor_candi = ["delta","loss"]
"""
delta : Maximum change in decomposed W and H
loss : Amount of loss change. It is derived from loss function.
"""

class NRBdMF():
    def __init__(self,cfix=1,K1=5,K2=5,num_factors=50,theta=1.0,lambda_d=0.625,lambda_t=0.625,alpha=0.1,beta=0.1,
                 max_iter=5000,tolx=1e-4,positive_weight=1,negative_weight=1,missing_base=0,monitor="loss",indicator=True,half_mask=False,verbose=False):
        self.cfix = cfix                            # weight to be imposed on the whole
        self.K1 = int(K1)                           # consider influences from the K1 nearest neighbors
        self.K2 = int(K2)                           # consider influences from the K2 nearest neighbors in prediction procedure
        self.num_factors = int(num_factors)         # dimensions of latent factors
        self.theta = float(theta)                   # learning rate
        self.lambda_d = float(lambda_d)             # Frobenius norm weight of drugs
        self.lambda_t = float(lambda_t)             # Frobenius norm weight of targets
        self.alpha = float(alpha)                   # weight of the drug's neighborhood regularization term
        self.beta = float(beta)                     # weight of the target's neighborhood regularization term
        self.max_iter = int(max_iter)               # maximum number of iterations
        self.tolx = tolx                            # tolerance (use for early stopping)
        self.positive_weight = positive_weight      # weight on positive labels
        self.negative_weight = negative_weight      # weight on negative labels
        self.missing_base = missing_base            # weight on missing labels
        self.monitor = monitor                      # how to monitor the algorithm update
        self.indicator = indicator                  # whether to put weight on the labels
        self.half_mask = half_mask                  # whether to consider missing labels as 0.5 or not
        self.verbose = verbose                      # whether to display the progress
        
#%% Basical modules
    def set_data(self,intMat,drugMat,targetMat):
        """
        Set data for analysis. 
        !!! All subsequent processes assume that the drugs are in rows and the targets are in columns. !!!
        ----------
        intMat : DataFrame
            A matrix that stores the known interactions to be analyzed.
            e.g. drugs in row and side-effects in column
            
                            Lactic acidosis ... Renal tubular acidosis
            methazolamide          0        ...         0
            abacavir               1        ...         0
            amphotericin           0        ...         1
            acebutolol             0        ...         0
            
        drugMat : DataFrame
            kerel for drugs.
            e.g. drug similarity matrix calculated from chemical structure.
        targetMat : DataFrame
            kernel for targets.
            e.g. disease similarity matrix defined by disease ontology.
        """
        self.intMat = np.array(intMat)
        self.drugMat = np.array(drugMat)
        self.targetMat = np.array(targetMat)
        self.num_drugs, self.num_targets = intMat.shape # number of drug and target respectively
        self.ones = np.ones((self.num_drugs, self.num_targets)) # matrix with all elements 1
        self.U = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_drugs, self.num_factors))
        self.V = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_targets, self.num_factors))
    
    def AGD_optimization(self, seed=None):
        """
        AdaGrad algorithm to accelerate the convergence of the gradient descent optimization.
        
        AdaGrad : Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
        - https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf

        """
        # setting initial values
        if seed is None:
            self.U = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_targets, self.num_factors))
        else:
            prng = np.random.RandomState(seed)
            self.U = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self.num_targets, self.num_factors))
        dg_sum = np.zeros((self.num_drugs, self.U.shape[1]))
        tg_sum = np.zeros((self.num_targets, self.V.shape[1]))
        sqrteps = np.sqrt(np.spacing(1))
        
        # displaying progress
        if self.verbose:
            pbar = tqdm(range(self.max_iter))
            last_log = self.calc_loss()
            for i in pbar:
                dg = self.deriv(True) # ∂J/∂U
                dg_sum += np.square(dg)
                vec_step_size = self.theta / np.sqrt(dg_sum)
                du = np.amax(abs(vec_step_size * dg) / (sqrteps+np.amax(abs(self.U))))
                self.U -= vec_step_size * dg # update the U
                tg = self.deriv(False) # ∂J/∂V
                tg_sum += np.square(tg)
                vec_step_size = self.theta / np.sqrt(tg_sum)
                dv = np.amax(abs(vec_step_size * tg) / (sqrteps+np.amax(abs(self.V))))
                self.V -= vec_step_size * tg # update the V
                
                delta = max(du,dv) # maximum change in decomposed W and H
                curr_log = self.calc_loss()
                loss = (curr_log-last_log)/abs(last_log) # amount of loss change
                pbar.set_postfix(DELTA=delta,LOSS=loss)
                last_log = curr_log # update the loss
                
                # set monitor
                if self.monitor == "delta":
                    monitar = delta
                elif self.monitor == "loss":
                    monitar = abs(loss)
                else:
                    raise ValueError("! Inappropriate monitaring target !")
                
                # determine early stopping by comparing the monit
                if monitar <= self.tolx:
                    print("--- meeted the tolx ---")
                    print(i,"/",self.max_iter,"iterations")
                    break
        # not displaying progress
        else:
            last_log = self.calc_loss()
            for i in range(self.max_iter):
                dg = self.deriv(True) # ∂J/∂U
                dg_sum += np.square(dg)
                vec_step_size = self.theta / np.sqrt(dg_sum)
                du = np.amax(abs(vec_step_size * dg) / (sqrteps+np.amax(abs(self.U))))
                self.U -= vec_step_size * dg # update the U
                tg = self.deriv(False) # ∂J/∂V
                tg_sum += np.square(tg)
                vec_step_size = self.theta / np.sqrt(tg_sum)
                dv = np.amax(abs(vec_step_size * tg) / (sqrteps+np.amax(abs(self.V))))
                self.V -= vec_step_size * tg # update the V
                
                delta = max(du,dv) # maximum change in decomposed W and H
                curr_log = self.calc_loss()
                loss = (curr_log-last_log)/abs(last_log) # amount of loss change
                last_log = curr_log # update the loss
                
                # set monitor
                if self.monitor == "delta":
                    monitar = delta
                elif self.monitor == "loss":
                    monitar = abs(loss)
                else:
                    raise ValueError("! Inappropriate monitaring target !")
                
                # determine early stopping by comparing the monit
                if monitar <= self.tolx:
                    break

    
    def deriv(self, drug):
        """
        partial differential function
        - ∂J/∂U  = R ° (UV^T  -Y)V+(λ_d I+ αL_d )U 
        - ∂J/∂V  =R^T  ° (VU^T  - Y^T )U+(λ_t I+ βL_t )V
        """
        # Weights are assigned to the input matrix values according to their labels.
        if self.indicator:
            posi_mask = np.where(self.intMat==1,self.positive_weight,0) # weight on positive labels
            nega_mask = np.where(self.intMat==-1,self.negative_weight,0) # weight on negative labels
            merge = posi_mask + nega_mask # merge positive and negative labels. Note that contradictory interactions are added up to 0
            merge_mask = np.where(merge==0,self.missing_base,merge)
            self.merge_mask = merge_mask
            
            if drug:
                # ∂J/∂U
                vec_deriv = np.dot(merge_mask*(np.dot(self.U,self.V.T) - self.intMat), self.V) # (UVt-Y)V
                vec_deriv += self.lambda_d*self.U+self.alpha*np.dot(self.DL, self.U)
            else:
                # ∂J/∂V
                vec_deriv = np.dot(merge_mask.T*(np.dot(self.V,self.U.T) - self.intMat.T), self.U) # (VUt-Yt)U
                vec_deriv += self.lambda_t*self.V+self.beta*np.dot(self.TL, self.V)
        # Use the input matrix values as they are.
        else:
            if drug:
                # ∂J/∂U
                vec_deriv = np.dot((np.dot(self.U,self.V.T) - self.intMat), self.V) # (UVt-Y)V
                vec_deriv += self.lambda_d*self.U+self.alpha*np.dot(self.DL, self.U)
            else:
                # ∂J/∂V
                vec_deriv = np.dot((np.dot(self.V,self.U.T) - self.intMat.T), self.U) # (VUt-Yt)U
                vec_deriv += self.lambda_t*self.V+self.beta*np.dot(self.TL, self.V)
        return vec_deriv
    
    def construct_neighborhood(self, drugMat, targetMat):
        """
        Define drug and target neighbors based on K1 values, respectively.

        Parameters
        ----------
        drugMat : DataFrame
            kerel for drugs.
            e.g. drug similarity matrix calculated from chemical structure.
        targetMat : DataFrame
            kernel for targets.
            e.g. disease similarity matrix defined by disease ontology.

        """
        self.dsMat = drugMat - np.diag(np.diag(drugMat))
        self.tsMat = targetMat - np.diag(np.diag(targetMat))
        if self.K1 > 0:
            S1 = self.get_nearest_neighbors(self.dsMat, self.K1)
            self.DL = self.laplacian_matrix(S1)
            S2 = self.get_nearest_neighbors(self.tsMat, self.K1)
            self.TL = self.laplacian_matrix(S2)
        else:
            self.DL = self.laplacian_matrix(self.dsMat)
            self.TL = self.laplacian_matrix(self.tsMat)
    
    def get_nearest_neighbors(self, S, size:int=5):
        """
        Subfunction used to define neighbors.

        Parameters
        ----------
        S : np.array
            Kernel minus its diagonal matrix.
        size : int
            Threshold as a neighborhood. The default is 5.
            
        Then, move on to the laplacian_matrix() 
        """
        m, n = S.shape
        X = np.zeros((m, n))
        for i in range(m):
            ii = np.argsort(S[i, :])[::-1][:min(size, n)]
            X[i, ii] = S[i, ii]
        return X
    
    def laplacian_matrix(self, S):
        """
        Create Laplacian matrix
        """
        x = np.sum(S, axis=0)
        y = np.sum(S, axis=1)
        L = 0.5*(np.diag(x+y) - (S+S.T))  # neighborhood regularization matrix
        return L

    
    def calc_loss(self):
        """
        Calculate loss based on the definition fo the loss function.
        """
        loss = 0
        tmp = self.intMat - np.dot(self.U,self.V.T)
        loss -= 0.5*np.sum(tmp*tmp)
        loss -= 0.5*self.lambda_d*np.sum(np.square(self.U)) + 0.5*self.lambda_t*np.sum(np.square(self.V))
        loss -= 0.5*self.alpha * np.sum(np.diag((np.dot(self.U.T, self.DL)).dot(self.U)))
        loss -= 0.5*self.beta * np.sum(np.diag((np.dot(self.V.T, self.TL)).dot(self.V)))
        return loss

#%% Practical modules
    def fix_model(self, W, intMat, drugMat, targetMat, seed=None):
        """
        Apply the model to the data. This is the main module to conduct NRBdMF.
        ----------
        W : np.array
            Mask matrix to separate train dataset and test dataset.
        intMat : DataFrame
            A matrix that stores the known interactions to be analyzed.
            e.g. drugs in row and side-effects in column
            
                            Lactic acidosis ... Renal tubular acidosis
            methazolamide          0        ...         0
            abacavir               1        ...         0
            amphotericin           0        ...         1
            acebutolol             0        ...         0
            
        drugMat : DataFrame
            kerel for drugs.
            e.g. drug similarity matrix calculated from chemical structure.
        targetMat : DataFrame
            kernel for targets.
            e.g. disease similarity matrix defined by disease ontology.
        seed : int, optional
            Bring reproducibility. The default is None.

        """
        intMat = np.array(intMat)
        drugMat = np.array(drugMat)
        targetMat = np.array(targetMat)
        self.W = W
        self.num_drugs, self.num_targets = intMat.shape # number of drug and target respectively
        self.ones = np.ones((self.num_drugs, self.num_targets)) # matrix with all elements 1
        
        if self.half_mask:
            rev_half = np.where(self.W == 1,0,0.5) # reverse the mask matrix
            self.intMat = self.cfix*intMat*self.W + rev_half # mask
        else:
            self.intMat = self.cfix*intMat*self.W # mask
        
        x, y = np.where(self.intMat > 0)
        self.train_drugs, self.train_targets = set(x.tolist()), set(y.tolist())
        self.construct_neighborhood(drugMat, targetMat)
        self.AGD_optimization(seed)
    
    def summarize(self):
        """
        Draw the estimated matrix as a heatmap
        """
        sns.heatmap(self.intMat)
        plt.title("original Y")
        plt.show()
        sns.heatmap(self.U.dot(self.V.T))
        plt.title("Y* ~ U V")
        plt.show()
    
    def inner_evaluation(self):
        """
        Evaluate the predicted value of the masked area under CVS1
        """
        score_mat = self.U.dot(self.V.T)
        scores = score_mat[~np.isnan(score_mat)] # same to np.array(pd.DataFrame(score_mat.flatten()).dropna())
        labels = self.intMat[~np.isnan(self.intMat)]
        self.pred_df = pd.DataFrame({"scores":scores,"label":labels}).astype({'scores':float,'label':int})

        posi_intMat = np.where(self.intMat == 1, 1, 0)
        nega_intMat = np.where(self.intMat == -1, 1, 0)
        posi_labels = posi_intMat[~np.isnan(posi_intMat)]
        nega_labels = nega_intMat[~np.isnan(nega_intMat)]
        
        calc_auc_aupr(posi_labels, scores, title="positive labels")
        calc_auc_aupr(nega_labels, [-1*t for t in scores], title="negative labels")
    
    def ex_evaluation(self, test_data, test_label):
        """
        Evaluate the predicted value of the masked area under CVS2 and CVS3
        """
        dinx = np.array(list(self.train_drugs))
        DS = self.dsMat[:, dinx]
        tinx = np.array(list(self.train_targets))
        TS = self.tsMat[:, tinx]
        scores = []
        # consider influences from the K2 nearest neighbors in prediction procedure
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
                scores.append(val)
        elif self.K2 == 0:
            for d, t in test_data:
                val = np.sum(self.U[d, :]*self.V[t, :])
                scores.append(val)
        # self.pred_res = pd.DataFrame({"scores":scores,"label":test_label}).sort_values("scores",ascending=False)
        self.pred_res = pd.DataFrame({"scores":scores,"label":test_label})
        posi_fxn = lambda x : 1 if x>=1 else 0 # convert label value : + 1--> +1, 0 --> 0, -1 --> 0
        posi_pred = copy.deepcopy(self.pred_res)
        posi_pred["label"] = self.pred_res["label"].apply(posi_fxn)
        nega_fxn = lambda x : 1 if x==-1 else 0 # convert label value : + 1--> -1, 0 --> 0, -1 --> 0
        nega_pred = copy.deepcopy(self.pred_res)
        nega_pred["label"] = self.pred_res["label"].apply(nega_fxn)
        
        # calculate AUROC and AUPR of positive label prediction
        posi_auroc, posi_aupr = calc_auc_aupr(posi_pred["label"].tolist(),posi_pred["scores"].tolist(),title="positive labels",verbose=self.verbose)
        # calculate AUROC and AUPR of negative label prediction
        nega_auroc, nega_aupr = calc_auc_aupr(nega_pred["label"].tolist(),[-1*t for t in nega_pred["scores"].tolist()],title="negative labels",verbose=self.verbose)
        return (posi_auroc,posi_aupr),(nega_auroc,nega_aupr)

def calc_auc_aupr(y_true:list,y_score:list,title="test",verbose=True):
    # calc each component of ROC
    try:
        fpr, tpr, thr = roc_curve(y_true, y_score)
        auroc = roc_auc_score(y_true, y_score)
        if verbose:
            # plot ROC
            plt.figure(figsize=(6,5))
            ax = plt.subplot(111)
            ax.plot(fpr, tpr, linewidth=2)
            plt.xlabel('FPR: False positive rate')
            plt.ylabel('TPR: True positive rate')
            plt.text(0.5,0.1,'AUROC = {}'.format(str(round(auroc,5))), transform=ax.transAxes, fontsize=15)
            plt.title(title)
            plt.show()
    
    # calc each component of PR
        prc, rec, thr2 = precision_recall_curve(y_true, y_score)
        aupr = auc(rec,prc)
        if verbose:
            plt.figure(figsize=(6,5))
            ax = plt.subplot(111)
            ax.plot(rec, prc, linewidth=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.text(0.5,0.8,'AUPR = {}'.format(str(round(aupr,5))), transform=ax.transAxes, fontsize=15)
            plt.title(title)
            plt.show()
    # If none of the positive (negative) labels exist, etc.
    except:
        print("something is wrong")
        auroc = np.nan
        aupr = np.nan
    
    return auroc, aupr