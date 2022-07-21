#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 10:41:01 2022

@author: docker
"""
from _utils import functions as fxn
from Representative_Methods.cv_eval_tolx_models import lmf_cv_eval
from Representative_Methods.cv_eval_tolx_models import nmf_cv_eval
from Representative_Methods.cv_eval_tolx_models import grnmf_cv_eval
from Representative_Methods.cv_eval_tolx_models import nrlmf_cv_eval
from Representative_Methods.cv_eval_tolx_models import cmf_cv_eval
from Representative_Methods.cv_eval_tolx_models import imc_cv_eval

class CV_tolx_Evaluator():
    def __init__(self,seeds = [7771, 8367, 22, 1812, 4659],folds=10,half_mask=False):
        self.seeds = seeds
        self.folds = int(folds)
        self.half_mask = half_mask
    
    def prepare(self,Y):
        self.Y = Y
        # CV setting CVS1
        cv = 1
        self.cv_data = fxn.cross_validation(self.Y, self.seeds, cv, num=self.folds)
        
    def set_data(self,Y,cv_data,drugMat=None,targetMat=None):
        self.Y = Y
        self.cv_data = cv_data
        self.drugMat = drugMat
        self.targetMat = targetMat
    
    def cv_optimize(self,method="LMF",eval_method="AUPR"):
        if method == "LMF":
            self.eval_opt, self.history = lmf_cv_eval(self.cv_data,self.Y,half_mask=self.half_mask)
            self.best_param = self.eval_opt[-1]
        elif method == "NMF":
            self.eval_opt, self.history = nmf_cv_eval(self.cv_data,self.Y,eval_method=eval_method,half_mask=self.half_mask)
            self.best_param = self.eval_opt[-1]
            self.history = self.history
        elif method == "GRNMF":
            self.eval_opt, self.history = grnmf_cv_eval(self.cv_data,self.Y,eval_method=eval_method,half_mask=self.half_mask)
            self.best_param = self.eval_opt[-1]
        elif method == "NRLMF":
            self.eval_opt, self.history = nrlmf_cv_eval(self.cv_data,self.Y,self.drugMat,self.targetMat,eval_method=eval_method,half_mask=self.half_mask)
            self.best_param = self.eval_opt[-1]
        elif method == "CMF":
            self.eval_opt, self.history = cmf_cv_eval(self.cv_data,self.Y,self.drugMat,self.targetMat,eval_method=eval_method,half_mask=self.half_mask)
            self.best_param = self.eval_opt[-1]
        elif method == "IMC":
            self.eval_opt, self.history = imc_cv_eval(self.cv_data,self.Y,self.drugMat,self.targetMat,eval_method=eval_method,half_mask=self.half_mask)
            self.best_param = self.eval_opt[-1]
        else:
            raise ValueError("!! In appropriate method !!")
        

