import argparse
import numpy as np
import json
import os
import cvxpy as cp
import scipy as sp
import datetime
from methods import Aggregator
from metal.label_model import LabelModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="../ama_logs/ama_final_runs")
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--model_prefix", type=str, default="EleutherAI_gpt-j-6B")
    parser.add_argument("--override_date", type=str, default=None)

    args = parser.parse_args()
    return args

def get_data(task_name, data_dir, model_prefix, override_date=None):
    """
    Load in dataset from task_name depending on where files are saved.
    """       
    task_dir = os.path.join(data_dir, task_name)
    date = datetime.datetime.today().strftime("%m%d%Y") if not override_date else override_date
    print(f"Loading runs from {date}")
    dpath = os.path.join(task_dir, f"{model_prefix}_decomposed_{date}.json")
    train_dpath = os.path.join(task_dir, f"{model_prefix}_decomposed_{date}_train.json")
    
    print(dpath)
    print(train_dpath)

    if task_name in ["super_glue_rte"]:
        label_name_to_int = {"True": 1, "False": 0}
    elif task_name == "amazon":
        label_name_to_int = {'amazon instant video':0,
                            'books':1,
                            'clothing shoes and jewelry':2,
                            'electronics':3,
                            'kindle store':4,
                            'movies and tv':5,
                            'musical instruments':6,
                            'office products':7,
                            'tools and home improvement':8} 

    elif task_name == 'wic':
        label_name_to_int = {"Yes": 1, "No": 0}
    elif task_name == 'super_glue_wsc':
        label_name_to_int = {"True": 1, "False": 0}
    elif task_name == "sst2":
        label_name_to_int = {"positive": 1, "negative": 0, "neutral": 0}
    elif task_name == "super_glue_boolq":
        label_name_to_int = {"true": 1, "false": 0}
    elif task_name in ["story_cloze", "story_cloze_v2", "story_cloze_v3"]:
        label_name_to_int = {"2": 1, "1": 0}
    elif task_name in ["anli_r1", "anli_r2", "anli_r3"]:
        label_name_to_int = {"true": 1, "false": 0, "neither": 2}
    elif task_name == "MR" or task_name == "mr":
        label_name_to_int = {"positive": 1, "negative": 0}
    elif task_name == "multirc":
        label_name_to_int = {"yes": 1, "no": 0}
    elif task_name == "super_glue_cb":
        label_name_to_int = {"true": 0, "false": 2, "neither": 1}
    elif task_name == "super_glue_copa":
        label_name_to_int = {"1": 1, "2": 0}
    elif task_name == "drop":
        label_name_to_int = {"true": 1, "false": 0}
    elif task_name == "super_glue_record":
        label_name_to_int = {"true": 1, "false": 0}
    elif task_name == "ag_news":
        label_name_to_int = {"world news": 0, "sports": 1, "business": 2, "technology and science": 3}
    elif task_name == "dbpedia":
        label_name_to_int = {"company": 0, "educational institution": 1, "artist": 2, "athlete": 3,
                             "office holder": 4, "mean of transportation": 5, "building": 6, "natural place": 7,
                             "village": 8, "animal": 9, "plant": 10, "album": 11, "film": 12, "written work": 13}
    else:
        raise ValueError("Unsupported task!")

    test_data = json.load(open(dpath))
    train_data = json.load(open(train_dpath))

    print(train_data['0']['example'])
    print(train_data['0']['gold'])

    n_test = len(test_data)
    n_train = len(train_data)

    m = len(test_data['0']['preds_boost'])

    test_votes = np.zeros((n_test, m))
    test_gold = np.zeros(n_test)
    for i in range(n_test):

        test_votes[i] = np.array([label_name_to_int[ans] if ans in label_name_to_int else -1 for ans in test_data[str(i)]['preds_boost']])
        test_gold[i] = label_name_to_int[str(test_data[str(i)]['gold'])]

    test_votes = test_votes.astype(int)
    test_gold = test_gold.astype(int)
    
    train_votes = np.zeros((n_train, m))
    train_gold = np.zeros(n_train)
    for i in range(n_train):
        train_votes[i] = np.array([label_name_to_int[ans] if ans in label_name_to_int else -1 for ans in train_data[str(i)]['preds_boost']])
        train_gold[i] = label_name_to_int[str(train_data[str(i)]['gold'])]

    train_votes = train_votes.astype(int)
    train_gold = train_gold.astype(int)

    return train_votes, train_gold, test_votes, test_gold



def get_top_deps_from_inverse_sig(J, k):
    m = J.shape[0]
    deps = []
    sorted_idxs = np.argsort(np.abs(J), axis=None)
    n = m*m 
    idxs = sorted_idxs[-k:]
    for idx in idxs:
        i = int(np.floor(idx / m))
        j = idx % m 
        if (j, i) in deps:
            continue
        deps.append((i, j))
    return deps

def learn_structure(L):
    m = L.shape[1]
    n = float(np.shape(L)[0])
    sigma_O = (np.dot(L.T,L))/(n-1) -  np.outer(np.mean(L,axis=0), np.mean(L,axis=0))
    
    #bad code
    O = 1/2*(sigma_O+sigma_O.T)
    O_root = np.real(sp.linalg.sqrtm(O))

    # low-rank matrix
    L_cvx = cp.Variable([m,m], PSD=True)

    # sparse matrix
    S = cp.Variable([m,m], PSD=True)

    # S-L matrix
    R = cp.Variable([m,m], PSD=True)

    #reg params
    lam = 1/np.sqrt(m)
    gamma = 1e-8

    objective = cp.Minimize(0.5*(cp.norm(R @ O_root, 'fro')**2) - cp.trace(R) + lam*(gamma*cp.pnorm(S,1) + cp.norm(L_cvx, "nuc")))
    constraints = [R == S - L_cvx, L_cvx>>0]

    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=False, solver=cp.SCS)
    opt_error = prob.value

    #extract dependencies
    J_hat = S.value
    
    if J_hat is None:
        raise ValueError("CVXPY failed to solve the structured learning problem, use result without dependencies.")
    
    for i in range(m):
        J_hat[i, i] = 0
    return J_hat


def learn_structure_multiclass(L, k):
    m = L.shape[1]
    J_hats = np.zeros((k, m, m))
    for c in range(k):

        all_votes_c = np.where(L == c, 1, 0)
        J_hats[c] = learn_structure(all_votes_c)

    return J_hats

def get_min_off_diagonal(J_hat):
    J_hat_copy = J_hat.copy()
    for i in range(len(J_hat_copy)):
        J_hat_copy[i, i] = np.inf
    return np.abs(J_hat_copy).min()

def main():
    args = get_args()
    task_name = args.task_name
    data_dir = args.data_dir
    symmetric = args.symmetric

    train_votes, train_gold, test_votes, test_gold = get_data(task_name, data_dir, args.model_prefix, args.override_date)
    classes = np.sort(np.unique(test_gold))
    vote_classes = np.sort(np.unique(test_votes))
    n_train, m = train_votes.shape
    n_test = len(test_votes)
    k = len(classes)
    abstains = len(vote_classes) == len(classes) + 1
    print(f"Abstains: {abstains}")

    m = test_votes.shape[1]

    all_votes= np.concatenate((train_votes, test_votes))

    label_model = LabelModel(k=k, seed=123)

    # scale to 0, 1, 2 (0 is abstain)
    test_votes_scaled = (test_votes + np.ones((n_test, m))).astype(int)
    test_gold_scaled = (test_gold + np.ones(n_test)).astype(int)

    train_votes_scaled = (train_votes + np.ones((n_train, m))).astype(int)
    train_gold_scaled = (train_gold + np.ones(n_train)).astype(int)

    all_votes_scaled = np.concatenate((train_votes_scaled, test_votes_scaled))

    label_model.train_model(all_votes_scaled, Y_dev=train_gold_scaled, abstains=abstains, symmetric=False, n_epochs=10000, log_train_every=50, lr=0.00001)


    print('Trained Label Model Metrics (No deps):')
    scores = label_model.score((test_votes_scaled, test_gold_scaled), metric=['accuracy','precision', 'recall', 'f1'])
    print(scores)

    all_votes_no_abstains = np.where(all_votes == -1, 0, all_votes)

    if len(classes) == 2:
        J_hat = learn_structure(all_votes_no_abstains)
    else:
        J_hats = learn_structure_multiclass(all_votes_no_abstains, len(classes))
        J_hat = J_hats.mean(axis=0)

    # if values in J are all too large, then everything is connected / structure learning isn't learning the right thing. Don't model deps then
    min_entry = get_min_off_diagonal(J_hat)
    if min_entry < 1:
        deps = get_top_deps_from_inverse_sig(J_hat, 1)
        print("Recovered dependencies: ", deps)

        label_model.train_model(all_votes_scaled, Y_dev=train_gold_scaled, abstains=abstains, symmetric=symmetric, n_epochs=80000, log_train_every=50, lr=0.000001, deps=deps)
        print('Trained Label Model Metrics (with deps):')
        scores = label_model.score((test_votes_scaled, test_gold_scaled), metric=['accuracy', 'precision', 'recall', 'f1'])
        print(scores)

    try:
        lm_probs = label_model.predict_proba(test_votes_scaled)
        agg = Aggregator(test_votes, test_gold, test_votes, test_gold, abstains, classes=[0, 1]) # 
        print("H(Y | WS output):")
        print(agg.conditional_entropy_singleton(lm_probs, test_gold))
    except:
        print(f"Failed to produce conditional entropy value: H(Y | WS output).")



if __name__ == "__main__":
    main()
