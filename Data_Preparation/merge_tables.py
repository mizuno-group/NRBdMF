#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 04:03:51 2022

@author: docker
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class MergeTables():
    def __init__(self):
        self.table1 = pd.DataFrame()
        self.table2 = pd.DataFrame()
        
    def set_tables(self,table1,table2):
        """
        Merge two matrices. Each matrix contains binary labels of 1 and 0.
        """
        self.table1 = table1
        self.table2 = table2
        fxn = lambda x : -1 if x == 1 else 0
        self.nega_table2 = table2.applymap(fxn)
    
    def merge(self,method="intersec"):
        if method == "intersec":
            # intersec
            tmp_merge = self.table1 + self.nega_table2
            intersec_merge = tmp_merge.dropna(how='all').dropna(how='all',axis=1)
            sns.heatmap(intersec_merge)
            plt.show()
            s = intersec_merge.sum().sum() / (len(intersec_merge)*len(intersec_merge.T))
            print("sparsity :",s)
            self.merged_table = intersec_merge
        elif method == "union":
            # union
            tmp_merge = self.table1 + self.nega_table2
            m_idx = tmp_merge.index.tolist()
            m_col = tmp_merge.columns.tolist()

            base = pd.DataFrame(index=m_idx,columns=m_col).fillna(0)
            ind_base = base.add(self.table1,fill_value=0)
            se_base = base.add(self.nega_table2,fill_value=0)

            union_merge = ind_base + se_base

            sns.heatmap(union_merge)
            plt.show()
            s = union_merge.sum().sum() / (len(union_merge)*len(union_merge.T))
            print("sparsity :",s)
            self.merged_table = union_merge
        else:
            print(["intersec","union"])
            raise ValueError("!! Inappropriate method. Select from above list !!")
    
    def detect_contradict(self):
        # union
        tmp_merge = self.table1 + self.table2
        m_idx = tmp_merge.index.tolist()
        m_col = tmp_merge.columns.tolist()

        base = pd.DataFrame(index=m_idx,columns=m_col).fillna(0)
        ind_base = base.add(self.table1,fill_value=0)
        se_base = base.add(self.table2,fill_value=0)

        union_merge_tmp = ind_base + se_base
        
        # detect the value == 2 (registered in both table1 and table2)
        fxn1 = lambda x : 1 if x==2 else 0
        union_merge = union_merge_tmp.applymap(fxn1)

        sns.heatmap(union_merge)
        plt.show()
        s = union_merge.sum().sum() / (len(union_merge)*len(union_merge.T))
        print("sparsity :",s)
        self.contradict_table = union_merge

