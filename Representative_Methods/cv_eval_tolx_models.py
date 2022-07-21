#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 10:24:33 2022

determine tolx with Cross Validation

@author: docker
"""
from _utils import functions as fxn
from Representative_Methods.methods import lmf, nmf, grnmf, nrlmf, cmf, imc_cv_eval
from tqdm import tqdm

def nmf_cv_eval(cv_data,intMat,eval_method="AUROC",half_mask=False):
    """Non-negative Matrix Factorization"""
    # Generate parameters
    params_grid = list()
    
    para = {'num_factors':50,'alpha':0.05,'tolx':1e-4,'max_iter':5000,'variance':0.01,'half_mask':half_mask}
    for r in [1,1e-1,1e-2,1e-3,1e-4]:
        params_grid.append({'num_factors':para['num_factors'],'alpha':para['alpha'],'tolx':r,'max_iter':para['max_iter'],'variance':para['variance'],'half_mask':para['half_mask']})
    
    # Grid Search
    max_auc, max_aupr, eval_opt = 0, 0, []
    history = []
    for param in tqdm(params_grid):
        model = nmf.DiegoNMF(**param)

        auc_avg, aupr_avg, auc_conf, aupr_conf = GridSearch(model,cv_data,intMat)
        info = [auc_avg, aupr_avg, auc_conf, aupr_conf, param]
        history.append(info)
        
        if eval_method == "AUROC":
            if auc_avg > max_auc:
                max_auc = auc_avg # update
                eval_opt = info # update
                print(param) # best parameter
                print("AUC :",auc_avg,"AUPR :",aupr_avg)
            else:
                pass
        elif eval_method == "AUPR":
            if aupr_avg > max_aupr:
                max_aupr = aupr_avg # update
                eval_opt = info # update
                print(param) # best parameter
                print("AUC :",auc_avg,"AUPR :",aupr_avg)
            else:
                pass
        else:
            raise ValueError("!! Inappropriate evaluation method !!")
    return eval_opt, history

def grnmf_cv_eval(cv_data,intMat,eval_method="AUROC",half_mask=False):
    """Graph Regularized Matrix Factorization"""
    # Generate parameters
    params_grid = list()
    
    para = {'num_factors':50,'max_iter':5000,'variance':0.01,'sigma':0.3,'tolx':1e-4,'half_mask':half_mask}
    for r in [1,1e-1,1e-2,1e-3,1e-4]:
        params_grid.append({'num_factors':para['num_factors'],'max_iter':para['max_iter'], 'variance':para['variance'], 'sigma':para['sigma'],'tolx':r,'half_mask':para['half_mask']})

    # Grid Search
    max_auc, max_aupr, eval_opt = 0, 0, []
    history = []
    for param in tqdm(params_grid):
        model = grnmf.GRNMF(**param)

        auc_avg, aupr_avg, auc_conf, aupr_conf = GridSearch(model,cv_data,intMat)
        info = [auc_avg, aupr_avg, auc_conf, aupr_conf, param]
        history.append(info)
        
        if eval_method == "AUROC":
            if auc_avg > max_auc:
                max_auc = auc_avg # update
                eval_opt = info # update
                print(param) # best parameter
                print("AUC :",auc_avg,"AUPR :",aupr_avg)
            else:
                pass
        elif eval_method == "AUPR":
            if aupr_avg > max_aupr:
                max_aupr = aupr_avg # update
                eval_opt = info # update
                print(param) # best parameter
                print("AUC :",auc_avg,"AUPR :",aupr_avg)
            else:
                pass
        else:
            raise ValueError("!! Inappropriate evaluation method !!")
    return eval_opt, history

def lmf_cv_eval(cv_data,intMat,eval_method="AUROC",half_mask=False):
    """Logistic Matrix Factorization"""
    # Generate parameters
    params_grid = list()
    
    para = {'num_factors':50, 'max_iter':5000, 'reg_param':0.6, 'gamma':1.0, 'tolx':1e-4, 'half_mask':half_mask}
    for r in [1,1e-1,1e-2,1e-3,1e-4]:
        params_grid.append({'num_factors':para['num_factors'],'max_iter':para['max_iter'],'reg_param':para['reg_param'],'gamma':para['gamma'],'tolx':r, 'half_mask':para['half_mask']})
    
    # Grid Search
    max_auc, max_aupr, eval_opt = 0, 0, []
    history = []
    for param in tqdm(params_grid):
        model = lmf.LMF(**param)

        auc_avg, aupr_avg, auc_conf, aupr_conf = GridSearch(model,cv_data,intMat)
        info = [auc_avg, aupr_avg, auc_conf, aupr_conf, param]
        history.append(info)
        
        if eval_method == "AUROC":
            if auc_avg > max_auc:
                max_auc = auc_avg # update
                eval_opt = info # update
                print(param) # best parameter
                print("AUC :",auc_avg,"AUPR :",aupr_avg)
            else:
                pass
        elif eval_method == "AUPR":
            if aupr_avg > max_aupr:
                max_aupr = aupr_avg # update
                eval_opt = info # update
                print(param) # best parameter
                print("AUC :",auc_avg,"AUPR :",aupr_avg)
            else:
                pass
        else:
            raise ValueError("!! Inappropriate evaluation method !!")
    return eval_opt, history

def nrlmf_cv_eval(cv_data,intMat,drugMat,targetMat,eval_method="AUROC",half_mask=False):
    """Neighborhood Regularized Logistic Matrix Factorization"""
    # Generate parameters
    params_grid = list()
    
    for r in [1,1e-1,1e-2,1e-3,1e-4]:
        params_grid.append({'cfix':5, 'K1':5, 'K2':5, 'num_factors':50, 'theta':1.0, 'lambda_d':0.625, 'lambda_t':0.625, 'alpha':0.1, 'beta':0.1, 'max_iter':5000, 'tolx':r, 'half_mask':half_mask})
    
    # Grid Search
    max_auc, max_aupr, eval_opt = 0, 0, []
    history = []
    for param in tqdm(params_grid):
        model = nrlmf.NRLMF(**param)

        auc_avg, aupr_avg, auc_conf, aupr_conf = GridSearch(model,cv_data,intMat,drugMat,targetMat)
        info = [auc_avg, aupr_avg, auc_conf, aupr_conf, param]
        history.append(info)
        
        if eval_method == "AUROC":
            if auc_avg > max_auc:
                max_auc = auc_avg # update
                eval_opt = info # update
                print(param) # best parameter
                print("AUC :",auc_avg,"AUPR :",aupr_avg)
            else:
                pass
        elif eval_method == "AUPR":
            if aupr_avg > max_aupr:
                max_aupr = aupr_avg # update
                eval_opt = info # update
                print(param) # best parameter
                print("AUC :",auc_avg,"AUPR :",aupr_avg)
            else:
                pass
        else:
            raise ValueError("!! Inappropriate evaluation method !!")
    return eval_opt, history

def cmf_cv_eval(cv_data,intMat,drugMat,targetMat,eval_method="AUROC",half_mask=False):
    """Collaborative Matrix Factorization"""
    # Generate parameters
    params_grid = list()
    
    for r in [1,1e-1,1e-2,1e-3,1e-4]:
        params_grid.append({'num_factors':50, 'lambda_l':0.01, 'lambda_d':0.01, 'lambda_t':0.01, 'max_iter':5000, 'tolx':r, 'half_mask':half_mask})
    
    # Grid Search
    max_auc, max_aupr, eval_opt = 0, 0, []
    history = []
    for param in tqdm(params_grid):
        model = cmf.CMF(**param)

        auc_avg, aupr_avg, auc_conf, aupr_conf = GridSearch(model,cv_data,intMat,drugMat,targetMat)
        info = [auc_avg, aupr_avg, auc_conf, aupr_conf, param]
        history.append(info)
        
        if eval_method == "AUROC":
            if auc_avg > max_auc:
                max_auc = auc_avg # update
                eval_opt = info # update
                print(param) # best parameter
                print("AUC :",auc_avg,"AUPR :",aupr_avg)
            else:
                pass
        elif eval_method == "AUPR":
            if aupr_avg > max_aupr:
                max_aupr = aupr_avg # update
                eval_opt = info # update
                print(param) # best parameter
                print("AUC :",auc_avg,"AUPR :",aupr_avg)
            else:
                pass
        else:
            raise ValueError("!! Inappropriate evaluation method !!")
    return eval_opt, history

def imc_cv_eval(cv_data,intMat,drugMat,targetMat,eval_method="AUROC",half_mask=False):
    """Inductive Matrix Completion"""
    # Generate parameters
    params_grid = list()
    
    for r in [1,1e-1,1e-2,1e-3,1e-4]:
        params_grid.append({'num_factors':10, 'reg_param1':1., 'reg_param2':1., 'variance':0.01, 'max_iter':5000, 'tolx':r, 'half_mask':half_mask})
    
    # Grid Search
    max_auc, max_aupr, eval_opt = 0, 0, []
    history = []
    for param in tqdm(params_grid):
        model = imc.IMC(**param)

        auc_avg, aupr_avg, auc_conf, aupr_conf = GridSearch(model,cv_data,intMat,drugMat,targetMat)
        info = [auc_avg, aupr_avg, auc_conf, aupr_conf, param]
        history.append(info)
        
        if eval_method == "AUROC":
            if auc_avg > max_auc:
                max_auc = auc_avg # update
                eval_opt = info # update
                print(param) # best parameter
                print("AUC :",auc_avg,"AUPR :",aupr_avg)
            else:
                pass
        elif eval_method == "AUPR":
            if aupr_avg > max_aupr:
                max_aupr = aupr_avg # update
                eval_opt = info # update
                print(param) # best parameter
                print("AUC :",auc_avg,"AUPR :",aupr_avg)
            else:
                pass
        else:
            raise ValueError("!! Inappropriate evaluation method !!")
    return eval_opt, history

def GridSearch(model,cv_data,intMat,drugMat=None,targetMat=None):
    """
    main function to evaluate each model
    """
    if drugMat is None:
        aupr_vec, auc_vec = fxn.train(model, cv_data, intMat)
    else:
        aupr_vec, auc_vec = fxn.train(model=model, cv_data=cv_data, intMat=intMat, drugMat=drugMat, targetMat=targetMat)
        
    aupr_avg, aupr_conf = fxn.mean_confidence_interval(aupr_vec)
    auc_avg, auc_conf = fxn.mean_confidence_interval(auc_vec)
    return auc_avg, aupr_avg, auc_conf, aupr_conf