# Compare the performance of seven representative algorithms
```
/Representative_Methods
│  cv_eval_tolx_models.py  # main module to conduct each method
│  cv_tolx_evaluator.py  # submodule used in main module
│  README.md  # this script
│
├─benchmark_datasets
│  ├─drug_disease  # called "Cdataset" and downloaded via http://github.com//bioinfomaticsCSU/MBiRW
│  │      disease_409_50_vec.csv
│  │      disease_409_gram_sim.csv
│  │      drdi.csv
│  │      drug_663_100_vec.csv
│  │      drug_663_gram_sim.csv
│  │
│  └─drug_protein
│          drug_sim.pkl
│          drug_vector_d100.txt
│          dtinet_708_db2cui_dic.pkl
│          mat_drug_protein_remove_homo.txt
│          protein_sim.pkl
│          protein_vector_d400.txt
│
├─methods # seven representative algorithms
│  │  cmf.py
│  │  grnmf.py
│  │  imc.py
│  │  lmf.py
│  │  nmf.py
│  │  nrlmf.py
│  │  tmf.py
│
└─__pycache__
        cv_eval_tolx_models.cpython-38.pyc
        cv_tolx_evaluator.cpython-38.pyc
```
