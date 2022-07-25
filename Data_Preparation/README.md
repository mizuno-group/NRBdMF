## Prepare multilabel data from SIDER
```
/Data_Preparation
│  merge_tables.py  # main module to create multilabel data
│  README.md  # this scrips
│  sider_rel_extraction_reflect_meddra.py  # module to extract drug effects from SIDER
│  __init__.py
│
├─MedDRA  # includes dictionary to convert MedDRA term to UMLS CUI
│  │  hlgt_name2cui.pkl
│  │  hlt_name2cui.pkl
│  │  pt_name2cui.pkl
│  │  soc_name2cui.pkl
│  │
│  └─structure
│          MedDRA_21_1.txt
│          MedDRA_21_1_CUI.txt
│
├─ver4_1  # SIDER 4.1: Side Effect Resource
│      meddra_all_indications.tsv
│      meddra_all_se.tsv
│      README.txt
│
└─__pycache__
        merge_tables.cpython-38.pyc
        sider_rel_extraction_reflect_meddra.cpython-38.pyc
        __init__.cpython-38.pyc
```
### Get Started
---
1. Extract side effects and indication information from SIDER 4.1 with `sider_rel_extraction_reflect_meddra.py`
2. Merge above dual nature of drug effects as positive and negative labels with `merge_tables.py`
   - positive labels : known side effects <br>
   - negative labels : know indications
