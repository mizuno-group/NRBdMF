
import os
import numpy as np
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import KFold,StratifiedKFold
from tqdm import tqdm


def load_data_from_file(dataset, folder):
    with open(os.path.join(folder, dataset+"_admat_dgc.txt"), "r") as inf:
        next(inf)
        int_array = [line.strip("\n").split()[1:] for line in inf]

    with open(os.path.join(folder, dataset+"_simmat_dc.txt"), "r") as inf:  # the drug similarity file
        next(inf)
        drug_sim = [line.strip("\n").split()[1:] for line in inf]

    with open(os.path.join(folder, dataset+"_simmat_dg.txt"), "r") as inf:  # the target similarity file
        next(inf)
        target_sim = [line.strip("\n").split()[1:] for line in inf]

    intMat = np.array(int_array, dtype=np.float64).T    # drug-target interaction matrix
    drugMat = np.array(drug_sim, dtype=np.float64)      # drug similarity matrix
    targetMat = np.array(target_sim, dtype=np.float64)  # target similarity matrix
    return intMat, drugMat, targetMat


def get_drugs_targets_names(dataset, folder):
    with open(os.path.join(folder, dataset+"_admat_dgc.txt"), "r") as inf:
        drugs = next(inf).strip("\n").split()
        targets = [line.strip("\n").split()[0] for line in inf]
    return drugs, targets


def read_params(file):
        keys = list()
        vals = list()
        for line in open(file,"r"):
            list_ = line.strip().split()
            if len(list_) == 0: continue
            keys.append(list_[0])
            vals.append(list(map(float,list_[1:])))

        grid = np.c_[tuple(map(np.ravel,np.meshgrid(*vals)))]
        params = list(map(lambda x: dict(zip(keys,x)),grid))
        
        return params

def cross_validation(intMat, seeds, cv=0, num=10, stratified=False):
    cv_data = defaultdict(list)
    intMat = np.array(intMat)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cv == 0:
            index = prng.permutation(num_drugs) # sort randomly
            step = int(index.size/num) # split into num sets
            for i in range(num):
                if i < num-1:
                    ii = index[i*step:(i+1)*step]
                else:
                    ii = index[i*step:]
                test_data = np.array([[k, j] for k in ii for j in range(num_targets)],dtype=np.int32)
                x, y = test_data[:, 0], test_data[:, 1]
                test_label = intMat[x, y]
                W = np.ones(intMat.shape)
                W[x, y] = 0 # mask the cells (row or column if cv == 0) corresponding to the test data
                cv_data[seed].append((W, test_data, test_label))
            
        elif cv == 1:
            total_labels = intMat.flatten() # flatten in row-major order
            label_index = np.array([i for i in range(len(total_labels))])
            
            # Stratified KFold
            if stratified:
                #print("---Stratified KFold---")
                skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
                for train_index,test_index in skf.split(label_index,total_labels):
                    #y_train, y_test = total_labels[train_index], total_labels[test_index]
                    #print(np.count_nonzero(y_test==1)) # check the stratified kfold work or not
                    test_data = np.array([[k/num_targets, k % num_targets] for k in test_index],dtype=np.int32)
                    
                    x, y = test_data[:, 0], test_data[:, 1]
                    test_label = intMat[x, y]
                    W = np.ones(intMat.shape)
                    W[x, y] = 0 # mask the cells (row or column if cv == 0) corresponding to the test data
                    cv_data[seed].append((W, test_data, test_label))
            # KFold
            else:
                #print("---KFold---")
                kf = KFold(n_splits=10,shuffle=True,random_state=seed)
                for train_index,test_index in kf.split(label_index,total_labels):
                    #y_train, y_test = total_labels[train_index], total_labels[test_index]
                    #print(np.count_nonzero(y_test==1)) # check the stratified kfold work or not
                    test_data = np.array([[k/num_targets, k % num_targets] for k in test_index],dtype=np.int32)
                    
                    x, y = test_data[:, 0], test_data[:, 1]
                    test_label = intMat[x, y]
                    W = np.ones(intMat.shape)
                    W[x, y] = 0 # mask the cells (row or column if cv == 0) corresponding to the test data
                    cv_data[seed].append((W, test_data, test_label))
    return cv_data

def external_validation(intMat, seeds, cv=1, num=10, num_fold=5):
    assert cv == 1
    ev_data = defaultdict(list)
    matrix = pd.DataFrame(intMat)
    rows, columns = np.array(matrix.index), np.array(matrix.columns)
    pairs = np.array([(r,c) for r in rows for c in columns])
    for seed in seeds:
        np.random.seed(seed=seed)
        elements = np.random.permutation(pairs)
        step = len(elements) // num
        fold_data = list()
        for i in range(num):
            if i < num-1: fold_data.append(elements[i*step:(i+1)*step])
            else: fold_data.append(elements[i*step:])
        fold_data = np.array(fold_data)
        for i in range(num):
            test_data = fold_data[i]
            x, y = test_data[:, 0], test_data[:, 1]
            test_label = intMat[x, y]
            W = np.ones(intMat.shape)
            W[x, y] = 0
            intMat_train = intMat.copy()
            intMat_train[x, y] = 0
            cv_data = cross_validation(intMat_train, [seed], cv=cv, num=num_fold)
            ev_data[seed].append([W, test_data, test_label, cv_data])
    return ev_data


def train(model, cv_data, intMat, drugMat=None, targetMat=None, cvs=1, double_split=False):
    """
    ----------
    model : 
        load from models in .Matrix_completion/XXX.py
    cv_data : defaultdict
        res from cross_validation()
    intMat : Array of float64
        interaction matrix (n x m)
    drugMat : Array of float64
        drug similarity matrix (n x k (n))
    targetMat : Array of float64
        target similarity matrix (m x k (m))
    cvs : int
        CVS1,2,3
    double_split : bool
        The default is False.

    """
    if drugMat is not None:
        if drugMat.min() == 0:
            drugMat = drugMat + np.spacing(1)
        if targetMat.min() == 0:
            targetMat = targetMat + np.spacing(1)
    
    aupr, auc = [], []
    if cvs == 1:
        for seed in cv_data.keys():
            for W, test_data, test_label in cv_data[seed]:
                if drugMat is None:
                    if targetMat is None:
                        model.fix_model(W, intMat, seed)
                else:
                    model.fix_model(W, intMat, drugMat, targetMat, seed)
                aupr_val, auc_val = model.evaluation(test_data, test_label)
                aupr.append(aupr_val)
                auc.append(auc_val)
    elif cvs == 2:
         for seed in cv_data.keys():
            for W, test_data, test_label in cv_data[seed]:
                if double_split:
                    """ NRLMF, CMF """
                    # train with full label matrix and narrowed drug matrix
                    model.fix_model(W, intMat, drugMat, targetMat, seed)
                    aupr_val, auc_val = model.ex_evaluation(test_data, test_label)
                else:
                    """ TMF, IMC """
                    train_idx = np.where(np.sum(W,axis=1)!=0.) # detect full row index
                    train_intMat = intMat[train_idx] # reshape the intMat
                    train_drugMat = drugMat[train_idx]
                    full_W = np.ones(train_intMat.shape)
                    # train with full label matrix and narrowed drug matrix
                    model.fix_model(full_W, train_intMat, train_drugMat, targetMat, seed)
                    
                    #aupr_val, auc_val = model.evaluation(test_data, test_label)
                    miss_idx = np.where(np.sum(W,axis=1)==0.) # detect missing row index
                    test_intMat = intMat[miss_idx]
                    test_drugMat = drugMat[miss_idx]
                    aupr_val, auc_val = model.ex_evaluation(test_intMat, test_drugMat, targetMat)
                #res = model.pred_res
                aupr.append(aupr_val)
                auc.append(auc_val)
    elif cvs == 3:
         for seed in cv_data.keys():
            for W, test_data, test_label in cv_data[seed]:
                if double_split:
                    """ NRLMF, CMF """
                    # train with full label matrix and narrowed drug matrix
                    model.fix_model(W, intMat.T, targetMat, drugMat, seed)
                    aupr_val, auc_val = model.ex_evaluation(test_data, test_label)
                else:
                    """ TMF, IMC """
                    train_idx = np.where(np.sum(W,axis=1)!=0.) # detect full row index
                    train_intMat = intMat.T[train_idx] # reshape the intMat
                    train_targetMat = targetMat[train_idx]
                    full_W = np.ones(train_intMat.shape)
                    # train with full label matrix and narrowed drug matrix
                    model.fix_model(full_W, train_intMat, train_targetMat, drugMat, seed)
                    
                    #aupr_val, auc_val = model.evaluation(test_data, test_label)
                    miss_idx = np.where(np.sum(W,axis=1)==0.) # detect missing row index
                    test_intMat = intMat.T[miss_idx]
                    test_targetMat = targetMat[miss_idx]
                    aupr_val, auc_val = model.ex_evaluation(test_intMat, test_targetMat, drugMat)
                #res = model.pred_res
                aupr.append(aupr_val)
                auc.append(auc_val)
    return np.array(aupr, dtype=np.float64), np.array(auc, dtype=np.float64)


def svd_init(M, num_factors):
    from scipy.linalg import svd
    U, s, V = svd(M, full_matrices=False)
    ii = np.argsort(s)[::-1][:num_factors]
    s1 = np.sqrt(np.diag(s[ii]))
    U0, V0 = U[:, ii].dot(s1), s1.dot(V[ii, :])
    return U0, V0.T


def mean_confidence_interval(data, confidence=0.95):
    import scipy as sp
    import scipy.stats
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


def write_metric_vector_to_file(auc_vec, file_name):
    np.savetxt(file_name, auc_vec, fmt='%.6f')


def load_metric_vector(file_name):
    return np.loadtxt(file_name, dtype=np.float64)
