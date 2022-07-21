#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 08:45:13 2022

@author: docker
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc, roc_auc_score

def matrix2pair(df,row_name="drug",col_name="disease",val_name="weight"):
    """ transform matrix to pair dataframe"""
    rec_f = np.array(df).flatten()
    new_x = []
    new_y = []
    for i in range(len(df)):
        new_x.extend([df.index.tolist()[i]]*len(df.T))
        new_y.extend(df.columns.tolist())
    res = pd.DataFrame({row_name:new_x,col_name:new_y,val_name:rec_f})
    res = res.sort_values(val_name,ascending=False)
    return res

def pair_compare(df,label_df,row_name="drug",col_name="disease",val_name="weight"):
    """ add origina true label col"""
    res1 = matrix2pair(df,row_name=row_name,col_name=col_name,val_name=val_name)
    res2 = matrix2pair(label_df)
    res1["label"]=res2["weight"]
    merge_res = pd.DataFrame(res1)
    merge_res = merge_res.sort_values(val_name,ascending=False)
    return merge_res

def triple_pair_compare(res_df,norm_df,original_df,row_name="drug",col_name="disease",val_name="weight"):
    """ add normalized true label and original true label columns"""
    res1 = matrix2pair(res_df,row_name=row_name,col_name=col_name,val_name=val_name)
    res2 = matrix2pair(norm_df,row_name=row_name,col_name=col_name,val_name=val_name)
    res3 = matrix2pair(original_df,row_name=row_name,col_name=col_name,val_name=val_name)
    res1["norm label"]=res2["weight"]
    res1["raw label"]=res3["weight"]
    merge_res = pd.DataFrame(res1)
    merge_res = merge_res.sort_values(val_name,ascending=False)
    return merge_res

def min_max(x,axis=None):
    m = x.min(axis=axis,keepdims=True)
    M = x.max(axis=axis,keepdims=True)
    result = (x-m)/(M-m)
    return result

def posi_signal(m,ntop=10):
    """
    detect top n positive signal
    """
    norm_m = min_max(m)
    threshold = sorted(norm_m.flatten(),reverse=True)[ntop]
    #posi = norm_m > threshold
    #print(threshold)
    posi = np.where(norm_m > threshold, 1, 0)
    return posi

def posi_signal2(m,threshold=0.5):
    """
    detect signal with hard threshold
    """
    norm_m = min_max(m)
    posi = norm_m > threshold
    return posi

def count_true_signal(y_test,posi):
    merge = y_test + posi
    posiposi = np.count_nonzero(merge==2)
    posinega = np.count_nonzero(y_test==1)-posiposi
    negaposi = np.count_nonzero(posi==1)-posiposi
    neganega = np.count_nonzero(merge==0)
    print("---result---")
    print("true=1, pred=1 :",posiposi)
    print("true=1, pred=0 :",posinega)
    print("true=0, pred=1 :",negaposi)
    print("true=0, pred=0 :",neganega)
    print("")
    print("accuacy :",posiposi/(posiposi + posinega))

def samesize_compare(y_test,pre,title1="y_test",title2="x_test W Ht X2 (prediction)"):
    """detect positive signal same number to the true label"""
    true_n = np.count_nonzero(y_test==1)
    posi_pre = posi_signal(pre,ntop=true_n)
    count_true_signal(y_test,posi_pre)
    # plot heatmap
    plot_dual(y_test,posi_pre,title1=title1,title2=title2)

# plot functions
def plot_dual(y_test,pre,title1="y_test",title2="x_test W Ht X2 (prediction)",cbar=False):
    """
    y_test : true label
    pre : prediction with x_test
    """
    # plot result
    fig,ax = plt.subplots(figsize=(15,6))
    plt.subplot(1,2,1)
    sns.heatmap(y_test,cbar=cbar)
    plt.title(title1)
    #plt.show()
    plt.subplot(1,2,2)
    sns.heatmap(pre,cbar=cbar)
    plt.title(title2)
    plt.show()

def plot_several(y_train,res,y_test,pre):
    """
    res : approximation with x_train
    pre : prediction with x_test
    """
    sns.heatmap(y_train)
    plt.title("y_train")
    plt.show()
    sns.heatmap(res)
    plt.title("x_train W Ht X2")
    plt.show()
    
    sns.heatmap(y_test)
    plt.title("y_test")
    plt.show()
    sns.heatmap(pre)
    plt.title("x_test W Ht X2")
    plt.show()

def plot_auroc_aupr(y_true:list,y_score:list,title="test"):
    # calc each component of ROC
    fpr, tpr, thr = roc_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    # plot ROC
    plt.figure(figsize=(6,5))
    ax = plt.subplot(111)
    plt.plot(fpr, tpr, linewidth=2)
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.text(0.5,0.1,'AUROC = {}'.format(str(round(auroc,5))), transform=ax.transAxes, fontsize=15)
    plt.title(title)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(color="#ababab",linewidth=0.5)
    #plt.legend(loc="best",shadow=True)
    plt.show()
    
    # calc each component of PR
    prc, rec, thr2 = precision_recall_curve(y_true, y_score)
    aupr = auc(rec,prc)
    # plot PR
    plt.figure(figsize=(6,5))
    ax = plt.subplot(111)
    plt.plot(rec, prc, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.text(0.5,0.8,'AUPR = {}'.format(str(round(aupr,5))), transform=ax.transAxes, fontsize=15)
    plt.title(title)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(color="#ababab",linewidth=0.5)
    #plt.legend(loc="best",shadow=True)
    plt.show()

def plot_aupr_auc_bar(aupr=[0.12,0.23,0.34],auroc=[0.78,0.89,0.90],name1="AUPR",name2="AUROC",value=" "):
    """
    ref : https://stats.biopapyrus.jp/python/barplot.html
    """
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('gist_yarg')
    
    fig,ax = plt.subplots()
    # bar
    df = pd.DataFrame({name1:aupr,name2:auroc})
    error_bar_set = dict(lw=1,capthick=1,capsize=20)
    ax.bar([0,1],df.mean(),yerr=df.std(),tick_label=df.columns,error_kw=error_bar_set)
    # jitter plot
    df_melt = pd.melt(df)
    sns.stripplot(x='variable', y='value', data=df_melt, jitter=True, color='black', ax = ax)
    
    ax.set_xlabel('')
    ax.set_ylabel(value)
    plt.show()

def enrichment_score(y_label:list,target=0,title="test"):
    """
    calculate enrichment score
    ----------
    y_label : list
        label outputs after sorting with score
    target : bool
        The default is 0. This means the negative label
    """
    up = 1/y_label.count(1)
    down = -1/(len(y_label) - y_label.count(1))
    
    res = []
    y = 0
    for t in y_label:
        if t == 1:
            y += up
        elif t == 0:
            y += down
        else:
            raise ValueError("! Inappropriate label !")
        res.append(y)
    
    # max - min
    wide = max(res) - min(res)
    # target statistics
    result = max(res) - abs(min(res))
    print('GSVA like statistics',round(result,5))
    
    fig, ax = plt.subplots()
    # plot barcode
    for i in range(len(y_label)):
        if y_label[i] == target:
            plt.vlines(i,0.0,-wide/8,colors='black',linewidth=1)
    plt.plot(res)
    plt.text(0.6, 0.95, 'GSVA statistics : {}'.format(str(round(result,5))), transform=ax.transAxes, fontsize=11)
    #plt.ylim()
    plt.xlabel('original rank in order')
    plt.ylabel('enrichment score')
    plt.title(title)
    plt.show()
    
def posi_nega_enrichment(y_label:list=[1,1,0,-1,1,0,0,-1,0,-1,-1],title="test",posi_label="side-effects",nega_label="indications",xlabel='original rank in order',ylabel='enrichment value',verbose=True):
    """
    calculate enrichment score
    ----------
    y_label : list
        label outputs after sorting with score
    """
    # positive label enrichment
    up = 1/y_label.count(1)
    down = -1/(len(y_label) - y_label.count(1))
    
    res1 = [0]
    y = 0
    for t in y_label:
        if t == 1:
            y += up
        else:
            y += down
        res1.append(y)
    
    # negative label enrichment
    up = 1/y_label.count(-1)
    down = -1/(len(y_label) - y_label.count(-1))
    
    res2 = [0]
    y = 0
    for t in y_label:
        if t == -1:
            y += up
        else:
            y += down
        res2.append(y)
    
    # max - min
    wide1 = max(res1) - min(res1)
    wide2 = max(res2) - min(res2)
    wide = max(wide1,wide2)
    # target statistics
    result1 = max(res1) - abs(min(res1))
    result2 = max(res2) - abs(min(res2))
    print('GSVA like statistics (positive)',round(result1,5))
    print('GSVA like statistics (negative)',round(result2,5))
    
    if verbose:
        fig, ax = plt.subplots()
        # plot barcode
        for i in range(len(y_label)):
            if y_label[i] == 1:
                plt.vlines(i+1,0.0,wide/8,colors="tab:orange",linewidth=0.2)
            elif y_label[i] == -1:
                plt.vlines(i,0.0,-wide/8,colors="tab:blue",linewidth=0.2)
        plt.plot(res1,color="tab:orange",label=posi_label)
        plt.plot(res2,color="tab:blue",label=nega_label)
        
        # plot baseline
        plt.plot([0]*(len(y_label)+1),linestyle="dashed",color="black")
        
        plt.text(0.55, 0.95, 'GSVA (positive) : {}'.format(str(round(result1,4))), transform=ax.transAxes, fontsize=11)
        plt.text(0.55, 0.90, 'GSVA (negative) : {}'.format(str(round(result2,4))), transform=ax.transAxes, fontsize=11)
        #plt.ylim()
        plt.xlabel('original rank in order')
        plt.ylabel('enrichment value')
        plt.title(title)
        plt.legend(loc="upper left")
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(color="#ababab",linewidth=0.5)
        plt.show()
    else:
        pass
    return result1, result2

def plot_multi(data=[[11,50,37,202,7],[47,19,195,117,74],[136,69,33,47],[100,12,25,139,89]],names = ["+PBS","+Nefopam","+Ketoprofen","+Cefotaxime"],value="ALT (U/I)",title="",rotation=0):
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('gist_yarg')
        
    fig,ax = plt.subplots()
    
    df = pd.DataFrame()
    for i in range(len(data)):
        tmp_df = pd.DataFrame({names[i]:data[i]})
        df = pd.concat([df,tmp_df],axis=1)
    error_bar_set = dict(lw=1,capthick=1,capsize=20/(len(data)-1))
    ax.bar([i for i in range(len(data))],df.mean(),yerr=df.std(),tick_label=df.columns,error_kw=error_bar_set)
    
    # jitter plot
    df_melt = pd.melt(df)
    if len(data) > 4:
        sns.stripplot(x='variable', y='value', data=df_melt, jitter=True, color='black', ax = ax, size=200/len(data[0]))
    else:
        sns.stripplot(x='variable', y='value', data=df_melt, jitter=True, color='black', ax = ax)
    plt.xticks(rotation=rotation)
    ax.set_xlabel('')
    ax.set_ylabel(value)
    plt.title(title)
    plt.show()