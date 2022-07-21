#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 08:41:48 2022

@author: docker
"""
import numpy as np
from scipy.stats import chi2_contingency



def calc_G2(df):
    chi2,p,dof,ex_table = chi2_contingency(df,lambda_="log-likelihood")
    """
    ex_df is same to...
    ex_df = np.zeros((len(table),len(table.T)))
    for i in range(len(row_sum)):
        for j in range(len(col_sum)):
            ex_df[i,j] += (row_sum[i]*col_sum[j]/total_sum)
    """
    total_sum = np.sum(df.sum())
    ob_table = np.array(df)
    g2_scores = 2 * total_sum * (np.log(ob_table/ex_table))
    return g2_scores

#*****************************************************************************#
"""memo"""
do_info = pd.read_pickle('/mnt/FAERS/workspace/Recommendation/Drug_Indication/220212_reflect_disease_ontology_similarity/data/6271_CUI_posi_dic.pkl')
do_cui = list(do_info.values())

whole_rel = pd.read_pickle('/mnt/FAERS/workspace/Recommendation/Drug_Indication/220128_preliminary/result/699_7015_rel_table.pkl')

# select triples containing target disease CUI
do_rel = whole_rel[whole_rel["OBJECT_CUI"].isin(do_cui)] # 145619/262803

final_drug = do_rel["SUBJECT_NAME"].unique().tolist() # 693
final_disease = do_rel["OBJECT_CUI"].unique().tolist() # 1632

# create table
table = np.zeros((len(final_drug),len(final_disease)),dtype=int)
for i in range(len(final_drug)):
    tmp_df = do_rel[do_rel["SUBJECT_NAME"]==final_drug[i]]
    tmp_ind = tmp_df["OBJECT_CUI"].tolist()
    for element in tmp_ind:
        idx = final_disease.index(element)
        table[i,idx]+=1

cont_table = pd.DataFrame(table,index=final_drug,columns=final_disease)
