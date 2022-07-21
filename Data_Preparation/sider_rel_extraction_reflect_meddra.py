# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:03:31 2022

Extract drug disease relationships from SIDER.
Determine the layer according to MedDRA structure.

1. extract side-effects
2. extract indications

@author: I.Azuma
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent # C:\github\NRMMF
#print(BASE_DIR)


class SiderSideEffectExtractor():
    def __init__(self):
        self.raw_df = None
    
    def set_raw(self,data=None):
        """
        1 & 2: STITCH compound ids (flat/stereo, see above)
        3: UMLS concept id as it was found on the label
        4: MedDRA concept type (LLT = lowest level term, PT = preferred term; in a few cases the term is neither LLT nor PT)
        5: UMLS concept id for MedDRA term
        6: side effect name
        
        All side effects found on the labels are given as LLT. Additionally, the PT is shown. There is at least one
        PT for every LLT, but sometimes the PT is the same as the LLT. LLTs are sometimes too detailed, and therefore
        you might want to filter for PT.
        """
        if data is None:
            data = pd.read_table(os.path.join(BASE_DIR, 'Data_Preparation/ver4_1/meddra_all_se.tsv'),names=["STITCH_flat","STITCH_stereo","label_ID","MedDRA_type","MedDRA_UMLS","se_name"])
        else:
            pass
        data.columns = ["STITCH_flat","STITCH_stereo","label_ID","MedDRA_type","MedDRA_UMLS","se_name"]
        self.raw_df = data
        unique_comp = self.raw_df["STITCH_flat"].unique().tolist()
        unique_se = self.raw_df["MedDRA_UMLS"].unique().tolist()
        print("registered compounds :",len(unique_comp))
        print("registered side-effects :",len(unique_se))
    
    def narrow_meddra_type(self,target:list=["SOC","HLGT","HLT","PT"],meddra_path=None):
        """
        narrow the records for the target MedDRA types
        """
        if meddra_path is None:
            meddra_txt = pd.read_table(os.path.join(BASE_DIR, 'Data_Preparation/MedDRA/structure/MedDRA_21_1_CUI.txt'),sep=" ",index_col=0)
        else:
            meddra_txt = pd.read_csv(meddra_path)
        
        target_cui = meddra_txt[target].melt()['value'].unique().tolist()
        print(len(target_cui),"terms were selcted")
        
        self.target_df = self.raw_df[self.raw_df["MedDRA_UMLS"].isin(target_cui)]
        self.target_comp = self.target_df["STITCH_flat"].unique().tolist()
        self.target_se = self.target_df["MedDRA_UMLS"].unique().tolist()
    
    def create_table(self):
        """
        create table
        """
        se_set = []
        self.cont_table = np.zeros((len(self.target_comp),len(self.target_se)),dtype=int)
        for i,k in tqdm(enumerate(self.target_comp)):
            tmp_df = self.target_df[self.target_df["STITCH_flat"]==self.target_comp[i]]
            tmp_se = tmp_df["MedDRA_UMLS"].unique().tolist()
            se_set.append(set(tmp_se))
            for s in tmp_se:
                idx = self.target_se.index(s)
                self.cont_table[i,idx] += 1
        self.se_dic = dict(zip(self.target_comp,se_set))
        self.cont_table = pd.DataFrame(self.cont_table,index=self.target_comp,columns=self.target_se)
        print("completed")

class SiderIndicationExtractor():
    def __init__(self):
        self.raw_df = None
        
    def set_raw(self,data=None):
        """
        1: STITCH compound id (flat, see above)
        2: UMLS concept id as it was found on the label
        3: method of detection: NLP_indication / NLP_precondition / text_mention
        4: concept name
        5: MedDRA concept type (LLT = lowest level term, PT = preferred term; in a few cases the term is neither LLT nor PT)
        6: UMLS concept id for MedDRA term
        7: MedDRA concept name
        
        All side effects found on the labels are given as LLT. Additionally, the PT is shown. There is at least one
        PT for every LLT, but sometimes the PT is the same as the LLT.
        """
        if data is None:
            data = pd.read_table(os.path.join(BASE_DIR, 'Data_Preparation/ver4_1/meddra_all_indications.tsv'),names=["STITCH_flat","label_ID","Method","concept name","MedDRA_type","MedDRA_UMLS","meddra_name"])
        else:
            pass
        data.columns = ["STITCH_flat","label_ID","Method","concept name","MedDRA_type","MedDRA_UMLS","meddra_name"]
        self.raw_df = data
        unique_comp = self.raw_df["STITCH_flat"].unique().tolist()
        unique_ind = self.raw_df["MedDRA_UMLS"].unique().tolist()
        print("registered compounds :",len(unique_comp))
        print("registered indications :",len(unique_ind))
        
    def narrow_meddra_type(self,target:list=["SOC","HLGT","HLT","PT"],meddra_path=None):
        """
        narrow the records for the target MedDRA types
        """
        if meddra_path is None:
            meddra_txt = pd.read_table(os.path.join(BASE_DIR, 'Data_Preparation/MedDRA/structure/MedDRA_21_1_CUI.txt'),sep=" ",index_col=0)
        else:
            meddra_txt = pd.read_csv(meddra_path)
        
        target_cui = meddra_txt[target].melt()['value'].unique().tolist()
        print(len(target_cui),"terms were selcted")
        
        self.target_df = self.raw_df[self.raw_df["MedDRA_UMLS"].isin(target_cui)]
        self.target_comp = self.target_df["STITCH_flat"].unique().tolist()
        self.target_indication = self.target_df["MedDRA_UMLS"].unique().tolist()
    
    def create_table(self):
        """
        create table
        """
        indication_set = []
        self.cont_table = np.zeros((len(self.target_comp),len(self.target_indication)),dtype=int)
        for i,k in tqdm(enumerate(self.target_comp)):
            tmp_df = self.target_df[self.target_df["STITCH_flat"]==self.target_comp[i]]
            tmp_se = tmp_df["MedDRA_UMLS"].unique().tolist()
            indication_set.append(set(tmp_se))
            for s in tmp_se:
                idx = self.target_indication.index(s)
                self.cont_table[i,idx] += 1
        self.ind_dic = dict(zip(self.target_comp,indication_set))
        self.cont_table = pd.DataFrame(self.cont_table,index=self.target_comp,columns=self.target_indication)
        print("completed")