## Input multilabel data for NRBdMF
```
/Data
│  README.md
│
├─646x499  # 646 drugs and 499 diseases
│      do_499x499_sim.csv
│      dtinet_646x646_sim.csv
│      memo.txt
│      se_ind_merge_table.csv
│
├─cvs_data  # processed data for cross validation
│      simple_CVS2_NRBdMF.pkl
│      simple_CVS2_NRLMF.pkl
│      simple_CVS3_NRBdMF.pkl
│      simple_CVS3_NRLMF.pkl
│
├─raw_rel  # extracted raw side effects and indications interaction matrices
│      ind_1423x2154_binary.csv
│      merge_1355x1625_multilabel.csv
│      se_1429x4138_binary.csv
│
└─tools  # tool to process data (e.g. annotation)
        memo.txt
        pt_name2cui.pkl
        sider_cid2name.pkl
```
