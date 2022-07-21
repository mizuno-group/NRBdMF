#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 01:03:40 2022

@author: docker
"""
import pandas as pd
import numpy as np
import glob
from scipy.sparse import linalg

def rwr(A,maxiter=100,restartProb = 0.50):
    """
    Random Walk with Restart
    A : numpy.array
        similarity matrix
    """
    n = len(A)
    # add self-edge to isolated nodes
    A = A + np.diag([int(t) for t in sum(A)==0])
    
    # nomalize the adjacency matrix
    col_sum = sum(A)
    P = A/col_sum
    
    # personalized PageRank
    restart = np.eye(n)
    Q = np.eye(n)
    for i in range(maxiter):
        Q_new = (1-restartProb)*np.dot(P,Q) + restartProb*restart
        delta = np.linalg.norm(Q - Q_new, "fro")
        Q = Q_new
        if delta < 1e-6:
            print("--- early stopping ---")
            print(delta)
            break
    return Q

def dca():
    # concat networks
    l = glob.glob('path/to/rwr/results/*.csv')
    
    Q = pd.DataFrame()
    for p in l:
        tQ = pd.read_csv(p,index_col=0)
        print(tQ.shape)
        Q = pd.concat([Q,tQ],axis=1)
    
    nnode = len(Q)
    alpha = 1/nnode # small positive constant
    
    # avoid taking logarithm of zero
    L = np.log(Q+alpha) - np.log(alpha) # diffusion state matrix
    
    X = np.dot(L,np.conjugate(L.T))
    #U,s,Vh = linalg.svd(X)
    U,s,Vh = linalg.svds(X,k=400,which='LM')
    
    res = U * np.sqrt(np.sqrt(s))
    
    return res