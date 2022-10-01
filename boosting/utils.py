import numpy as np
import itertools

def get_probabilties(num_lfs, num_examples, predictions, label_name_to_int):
        
        lf_array = np.zeros((num_lfs, num_examples))
        golds = []

        # Collect golds and preds
        for i, (k, item) in enumerate(predictions.items()):
            preds = item['chosen_answers_lst']
            preds_mapped = []
            for p in preds:
                if p in label_name_to_int:
                    preds_mapped.append(label_name_to_int[p])
                else:
                    preds_mapped.append(0)
            preds = preds_mapped.copy()
            for lf_num, p in zip(range(num_lfs), preds):
                lf_array[lf_num][i] = p
            gold = label_name_to_int[item['gold']]
            golds.append(gold)
        golds = np.array(golds)
        neg_indices, pos_indices = [np.where(golds == -1)[0], np.where(golds == 1)[0]]
        indices = {
            -1: neg_indices,
            1: pos_indices
        }

        # [i, j, k] = Pr(prompt_i = j| y = k)
        # Accuracies
        lf_accuracies = []
        for i in range(num_lfs):
            lf_accuracies.append(np.sum(golds  == np.array(lf_array[i]))/num_examples)
        print(f"LF Accs: {lf_accuracies}")

        # [i, j, k] = Pr(prompt_i = j| y = k)
        classes = label_name_to_int.values()
        accs = np.zeros((num_lfs, len(classes), len(classes))) 
        for p in range(num_lfs):
            for i in classes:
                for j in classes:
                    j_idx = j
                    if j == -1:
                        j_idx = 0
                    i_idx = i
                    if i == -1:
                        i_idx = 0
                    accs[p, i_idx, j_idx] = len(np.where(lf_array[p, indices[i]] == j)[0]) / len(indices[i])

        # Compute probabilities
        pos_probs = []
        for i in range(num_lfs):
            sub_preds = lf_array[i][pos_indices]
            sub_golds = golds[pos_indices]
            pos_probs.append(np.sum(sub_golds  == np.array(sub_preds))/len(pos_indices))
        print(f"Pos Probs: {pos_probs}")

        neg_probs = []
        for i in range(num_lfs):
            sub_preds = lf_array[i][neg_indices]
            sub_golds = golds[neg_indices]
            neg_probs.append(np.sum(sub_golds  == np.array(sub_preds))/len(neg_indices))
        print(f"Neg Probs: {neg_probs}\n\n") 
        
        return lf_accuracies, accs, pos_probs, neg_probs, golds, indices
    
    
""" Independence Assumption: take the product of probabilities as p(L1, L2, ..., LK | y) """

# Pr(y = 1 | lf votes)
def get_cond_probs(votes, y, indices_train, golds_train, accs_train, num_lfs_test):
    prop_pos =  len(indices_train[1])/len(golds_train)
    pr_y = prop_pos if y == 1 else 1 - prop_pos
    prod = pr_y
    for i in range(num_lfs_test):
        if y == -1:
            y = 0
        prod *= accs_train[i, y, votes[i]]
    return prod 

# Pr(y = 1 | lf votes)
def get_probs(votes, indices_train, golds_train, acc_train, num_lfs_test):
    votes = [max(v, 0) for v in votes]
    numerator =  get_cond_probs(votes, 1, indices_train, golds_train, acc_train, num_lfs_test) 
    denominator = numerator + get_cond_probs(votes, -1, indices_train, golds_train, acc_train, num_lfs_test)
    return numerator / denominator


def get_nb_accuracy(num_examples_test, num_lfs_test, predictions_test, label_name_to_int, golds_test, indices_train, golds_train, accs_train):
    output = np.zeros(num_examples_test) 
    errors = 0
    for i, (k, item) in enumerate(predictions_test.items()):
        votes = item['chosen_answers_lst']
        votes_mapped = []
        for v in votes:
            if v in label_name_to_int:
                votes_mapped.append(label_name_to_int[v])
            else:
                votes_mapped.append(0)
        votes = votes_mapped.copy()
        probs =  np.round(get_probs(votes, indices_train, golds_train, accs_train, num_lfs_test))
        output[i] = probs

        # Mean squared error
        g = golds_test[i]
        if golds_test[i] == -1:
            g = 0
        error = np.abs(output[i] - g)**2
        errors += error
    accuracy = 1 - (errors / num_examples_test)
    return accuracy, output


def estimate_matrix(m, n, L):
    E_prod = np.zeros((m, m))
    l_avg = np.zeros(m)
    for i in range(n):
        l = L[i, :]
        l_avg += l
        E_prod += np.outer(l, l)
    
    l_avg = l_avg/n
    E_prod = E_prod/n
    
    cov = E_prod - np.outer(l_avg, l_avg)
    
    return (E_prod, cov, l_avg)
 

def get_vote_vectors(num_samples, num_lfs, predictions, label_name_to_int):
    vectors = np.zeros((num_samples, num_lfs+1), float)
    vectors_no_y = np.zeros((num_samples, num_lfs), float)
    labels_vector = np.zeros((num_samples, 1), float)
    for i, p in enumerate(predictions.values()):
        votes = p['chosen_answers_lst']
        votes_mapped = []
        for v in votes:
            if v in label_name_to_int:
                votes_mapped.append(label_name_to_int[v])
            else:
                votes_mapped.append(0)
        votes = votes_mapped.copy()
        # votes = [max(v, 0) for v in votes]
        gold = p['gold']
        gold = label_name_to_int[gold]
        vectors_no_y[i] = np.array(votes)
        vectors[i] = np.array(votes + [gold]) #- lf_accuracies_train
        labels_vector[i] = np.array([gold])
    print(f"Shape: {vectors.shape}")
    print(f"Sample: {vectors[0]}")
    
    return vectors, vectors_no_y, labels_vector

def get_feature_vector(vote_vectors, include_pairwise=False, include_singletons=True):
    feature_vectors = []
    for votes in vote_vectors:
        if include_singletons:
            feature_vector = list(votes[:])
        else:
            feature_vector = []
        if include_pairwise:
            for subset in itertools.combinations(votes[:], 2):
                feature_vector.append(subset[0] * subset[1])
        feature_vectors.append(feature_vector)
    X = np.matrix(feature_vectors)
    return X