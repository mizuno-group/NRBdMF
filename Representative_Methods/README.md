## Compare the performance of seven representative algorithms
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

### Get Started
***
#### Benchmark Datasets
- drug-protein (708 drugs and 1512 proteins)
  - drug similarity: chemical structure defined in https://github.com/luoyunan/DTINet.git by Luo.
  - protein similarity: based on primary sequences of proteins defined in https://github.com/luoyunan/DTINet.git by Luo.
- drug-disease
  - drug similarity: chemical structure defined in https://github.com/luoyunan/DTINet.git by Luo.
  - disease similarity: based on disease ontology defined in https://github.com/LiangXujun/LRSSL.git by Liang.


Following Luo’s method, the desired vectors were obtained by performing random walk with restart (RWR) and diffusion component analysis (DCA). As the number of dimensions for the low-dimensional vectors, 100 dimensions were selected for drugs in both benchmark datasets, 400 dimensions for proteins, and 50 dimensions for diseases.


#### Representative Algorithms
Representative_Methods/methods includes 7 representative algorithms that are rich in derivation throughout the survey.
1. nmf (non-negative matrix factorization)
2. grnmf (graph regularized matrix factorization)
3. lmf (logistic matrix factorization)
4. nrlmf (neighborhood regularized matrix factorization)
5. cmf (collective matrix factorization)
6. tmf (triple matrix factorization)
7. imc (inductive matrix facorization)
