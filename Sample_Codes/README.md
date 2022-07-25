## Sample Codes
Basic use of this repository.

### 1_representative_methods_grid_search.ipynb
- We compared the prediction performance of seven representative algorithms on benchmark datasets.
- The performances were evaluated with five times of 10-fold cross-validation (CV) under three different settings as named CVS1, CVS2 and CVS3. 
- A example of NRLMF under CVS1 is discribed.
```
Eval = cv_tolx_evaluator.CV_tolx_Evaluator()
Eval.set_data(Y=Y, cv_data=cvs1_data,drugMat=drugMat,targetMat=targetMat) # set cvs1,cvs2 or cvs3
Eval.cv_optimize(method="NRLMF",eval_method="AUPR") # set algorithm name and evluation method
```

### 2_multilabel_data_preparation.ipynb
- Prepare the multilabel matrix used in side effects prediction with NRBdMF.
- Note that labels of +1 and -1 simply mean the relationship between side effects and indications, respectively, while the label 0 is the missing value or the contradictory relationship where both side effects (+1 label) and indications (-1 label) are registered in SIDER and add up to 0.
- 646 drugs and 499 diseases, including 17546 positive labels and 2343 negative labels.

### 3_nrbdmf_vs_nrlmf_cvs2.ipynb
- Compare the prediction performance with 10-fold cross validation under CVS2.
- Our proposed NRBdMF shows ideal result that the positive labels are enriched, and the negative labels are sparse in the top ranked predictions.

### 4_nrbdmf_vs_nrlmf_cvs3.ipynb
- Compare the prediction performance with 10-fold cross validation under CVS3.
- Our proposed NRBdMF shows ideal result that the positive labels are enriched, and the negative labels are sparse in the top ranked predictions.

### 5_nrbdmf_case_study.ipynb
1. Prediction of candidate compounds that cause specific side effects (Hypertension)
2. Prediction of candidate side effects that would be caused by a specific compound (doxorubicin)
