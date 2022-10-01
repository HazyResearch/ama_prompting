"""This script contains code to execute different methods"""
from readline import append_history_file
from sklearn.metrics import accuracy_score
import numpy as np
from snorkel.labeling.model import LabelModel
from snorkel.utils import probs_to_preds
import itertools 
import math
import torch

import collections

from sklearn.linear_model import LogisticRegression

import networkx as nx


class Aggregator():

    def __init__(self, train_votes, train_gold, test_votes, test_gold,  abstains = False, classes=[0, 1], abstain_value = -1) -> None:
        # set votes and golds
        self.train_votes = train_votes 
        self.train_gold = train_gold
        self.test_votes = test_votes
        self.test_gold = test_gold 


        self.n_train, self.m = train_votes.shape 
        self.n_test = len(test_gold)


        # in some cases, we need a validation set split from the training data
        np.random.seed(0)
        indices = np.random.permutation(self.n_train)
        n_val = int(self.n_train / 5) # 20% of the training dataset
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        self.train_no_val_votes = self.train_votes[train_idx, :]
        self.val_votes = self.train_votes[val_idx, :]

        self.train_no_val_gold = self.train_gold[train_idx]
        self.val_gold = self.train_gold[val_idx]


        # check classes
        self.classes = classes
        self.k = len(classes)

        # print(np.unique(self.train_gold))
        # print(np.unique(classes))
        assert len(np.unique(self.train_gold)) == self.k 
        assert len(np.unique(self.test_gold)) == self.k 

        # check if abstains
        self.abstains = abstains
        #assert len(np.unique(self.train_votes)) == int(abstains) + self.k 
        #assert len(np.unique(self.test_votes)) == int(abstains) + self.k 
        self.abstain_value = abstain_value
        self.vote_classes = self.classes.copy()
        if abstains:
            assert self.abstain_value in self.train_votes 
            assert self.abstain_value in self.test_votes
            self.vote_classes.append(self.abstain_value)

        
        

        self.nodes = np.arange(self.m)

        # construct scaled arrays (for binary)
        self._get_scaled()

        # get true accuracies on train and test
        self._get_train_acc()
        self._get_test_acc()

        # estimate some parameters
        self._estimate_balance()
        self._estimate_coverage()
        self._estimate_accs()
        self._estimate_test_accs()
        self._estimate_symmetric_accs()
        self._estimate_fs_accs()



    def _get_scaled(self):
        """
        For binary tasks defined with classes [0, 1] and abstain -1, we construct scaled versions with classes [-1, 1] and abstain 0.
        Scaled versions of the data are used as input to certain methods that assume an Ising model (such as FlyingSquid).
        """
        if self.classes == [0, 1]:
            self.train_votes_scaled = 2*self.train_votes - 1
            self.test_votes_scaled = 2*self.test_votes - 1
            self.train_no_val_votes_scaled = 2*self.train_no_val_votes - 1
            self.val_votes_scaled = 2*self.val_votes - 1

            self.train_gold_scaled = 2*self.train_gold - 1
            self.test_gold_scaled = 2*self.test_gold - 1
            self.train_no_val_gold_scaled = 2*self.train_no_val_gold - 1
            self.val_gold_scaled = 2*self.val_gold - 1


            if self.abstains:
                self.train_votes_scaled[self.train_votes == self.abstain_value] = 0
                self.test_votes_scaled[self.test_votes == self.abstain_value] = 0
                self.train_no_val_votes_scaled[self.train_no_val_votes == self.abstain_value] = 0
                self.val_votes_scaled[self.val_votes == self.abstain_value] = 0

        else:
            self.train_votes_scaled = self.train_votes
            self.test_votes_scaled = self.test_votes
            self.train_no_val_votes_scaled = self.train_no_val_votes
            self.val_votes_scaled = self.val_votes


            self.train_gold_scaled = self.train_gold
            self.test_gold_scaled = self.test_gold
            self.train_no_val_gold_scaled = self.train_no_val_gold
            self.val_gold_scaled = self.val_gold


    def _set_clique_tree(self, edgeset):
        """
        Constructs a data structure c_tree that contains nodes and edges of the junction tree.

        Args:
            edgeset: List of tuples (i, j) for i, j = {0, ..., m}
        """
        G1 = nx.Graph()
        G1.add_nodes_from(self.nodes)
        G1.add_edges_from(edgeset)

        self.higher_order = len(edgeset) != 0

        
        # Check if graph is chordal
        # TODO: Add step to triangulate graph if not
        if not nx.is_chordal(G1):
            raise nx.NetworkXError("Graph triangulation not implemented.")

        # Create maximal clique graph G2
        # Each node is a maximal clique C_i
        # Let w = |C_i \cap C_j|; C_i, C_j have an edge with weight w if w > 0
        G2 = nx.Graph()
        for i, c in enumerate(nx.chordal_graph_cliques(G1)):
            G2.add_node(i, members=c)
        for i in G2.nodes():
            for j in G2.nodes():
                S = G2.nodes[i]["members"].intersection(G2.nodes[j]["members"])
                w = len(S)
                if w > 0:
                    G2.add_edge(i, j, weight=w, members=S)

        self.c_tree = nx.maximum_spanning_tree(G2) # should be maximum??? Because we want maximal separator sets
        # Return a minimum spanning tree of G2

    def _set_clique_data(self):
        """
        Creates a data structure c_data which maps cliques and separator sets to their maximal clique.
        """
        self.c_data = dict()
        for i in range(self.m):
            self.c_data[i] = {
                "vertices": [i],
                "max_cliques": set( # which max clique i belongs to
                    [
                        j
                        for j in self.c_tree.nodes()
                        if i in self.c_tree.nodes[j]["members"]
                    ]
                ),
            }

        # Get the higher-order clique statistics based on the clique tree
        # First, iterate over the maximal cliques (nodes of c_tree) and
        # separator sets (edges of c_tree)
        if self.higher_order:
            counter = 0
            for item in itertools.chain(self.c_tree.nodes(), self.c_tree.edges()):
                if isinstance(item, int):
                    C = self.c_tree.nodes[item]
                    C_type = "node"
                elif isinstance(item, tuple):
                    C = self.c_tree[item[0]][item[1]]
                    C_type = "edge"
                else:
                    raise ValueError(item)
                members = list(C["members"])
                nc = len(members)

                # Else add one column for each possible value
                if nc != 1:
                    # Add to self.c_data as well
                    #idx = counter + m
                    self.c_data[tuple(members)] = {
                        "vertices": members,
                        "max_cliques": set([item]) if C_type == "node" else set(item),
                    }
                    counter += 1


    def _estimate_balance(self):
        """ Estimates the class balance Pr(y) on training data. Sets self.balance to be an array of length k.
        """
        self.gold_idxs = [np.where(self.train_gold == c)[0] for c in self.classes]
        self.balance = np.array([len(self.gold_idxs[c])/self.n_train for c in range(self.k)])

    def _estimate_accs(self):
        """ Computes Pr(vote_i | y) on training data. Each prompt has k x k values.
        We ignore the abstaining case Pr(vote_i = 0 | y), since this is handled by self.coverage.
        """

        k_votes = self.k
        vote_classes = self.classes

        self.nb_accs = np.zeros((self.m, self.k, k_votes)) # [i, j, k] = Pr(prompt_i = j| y = k)
        for p in range(self.m):
            for i in range(self.k):
                for j in range(k_votes):
                    vc = vote_classes[j]
                    self.nb_accs[p, i, j] = len(np.where(self.train_votes[self.gold_idxs[i], p] == vc)[0]) / len(self.gold_idxs[i])

        # clip values to 0.0001, 0.9999
        self.nb_accs[self.nb_accs > 1] = 0.9999
        self.nb_accs[self.nb_accs == 0] = 0.0001

    def _estimate_test_accs(self):

        self.gold_test_idxs = [np.where(self.test_gold == c)[0] for c in self.classes]


        self.nb_test_accs = np.zeros((self.m, self.k, self.k)) # [i, j, k] = Pr(prompt_i = j| y = k)
        for p in range(self.m):
            for i in range(self.k):
                for j in range(self.k):
                    vc = self.classes[j]
                    self.nb_test_accs[p, i, j] = len(np.where(self.test_votes[self.gold_test_idxs[i], p] == vc)[0]) / len(self.gold_test_idxs[i])

        # clip values to 0.0001, 0.9999
        self.nb_test_accs[self.nb_test_accs > 1] = 0.9999
        self.nb_test_accs[self.nb_test_accs == 0] = 0.0001


    def _estimate_symmetric_accs(self):
        """ Computes Pr(vote_i | y) on training data similarly to above, but assumes Pr(vote_i = c | y = c) = Pr(vote_i = y),
        independent of what the value of y is. Then, Pr(vote_i = c' | y = c) = (1 - Pr(vote_i = y)) / (k - 1) (uniform assumption)
        """
        self.sym_accs = np.zeros((self.m, self.k, self.k))
        for i in range(self.m):
            for j in range(self.k):
                for k in range(self.k):
                    if j == k:
                        self.sym_accs[i, j, k] = self.train_acc[i] # Pr(lf_i = c | y = c) = Pr(lf = y)
                    else:
                        self.sym_accs[i, j, k] = (self.coverage[i] - self.train_acc[i])/(self.k - 1) # divide uniformly among other classes

            

    def _estimate_coverage(self):
        """ Computes Pr(vote_i != 0) (coverage) and Pr(vote_i = 0 | y) for each y (abstain_rate).       
        """
        # Pr(vote_i != 0)
        self.coverage = np.array([len(np.where(self.train_votes[:, p] != self.abstain_value)[0]) / self.n_train for p in range(self.m)])

        # Pr(vote_i = 0 | y)
        self.abstain_rate = np.zeros((self.m, self.k)) 
        for i in range(self.m):
            for j in range(self.k):
                self.abstain_rate[i, j] = len(np.where(self.train_votes[self.gold_idxs[j], i] == self.abstain_value)[0]) / len(self.gold_idxs[j])

        


    def _estimate_fs_accs(self, on_all_data = True):
        """ Estimates Pr(vote_i | y = 0, 1) using FlyingSquid algorithm. 
        Args:
        - on_test: If we use the unlabeled test dataset or the labeled train dataset. Default is True.

        This version of FlyingSquid only handles the binary case (and is called on one-vs-all for multiclass) and works with scaled data.
        """

        if self.k > 2:
            return

        if on_all_data:
            votes = np.concatenate((self.train_votes_scaled, self.test_votes_scaled))
            n = self.n_train + self.n_test
        else:
            votes = self.test_votes_scaled
            n = self.n_test 

        if self.abstains:
            # compute M[i, j] = E[vote_i * vote_j | vote_i, vote_j not abstaining]
            M = np.zeros((self.m, self.m))
            for i in range(self.m):
                for j in range(self.m):
                    no_abstains = np.where(np.logical_and(votes[:, i] != 0, votes[:, j] != 0))[0]
                    M[i, j] = votes[no_abstains, i].dot(votes[no_abstains, j]) / len(no_abstains)
        else:
            # M[i, j] = E[vote_i * vote_j]
            M = votes.T.dot(votes)/n
        triplets = list(itertools.combinations(np.arange(self.m), 3)) # all possible combinations of triplets

        self.fs_accs = np.zeros((self.m, 2, 2))
        total = math.comb(self.m-1, 2)
        # average over every combination of triplets
        for (i, j, k) in triplets:
            a = np.zeros(3)
            a[0] = 0.5*(np.sqrt(np.abs(M[i, j] * M[i, k] / M[j, k]))+1)
            a[1] = 0.5*(np.sqrt(np.abs(M[j, k] * M[i, j] / M[i, k]))+1)
            a[2] = 0.5*(np.sqrt(np.abs(M[i, k] * M[j, k] / M[i, j]))+1)

            # edge cases
            a[np.where(np.isnan(a))[0]] = 0.5 
            a[np.where(np.isinf(a))[0]] = 1

            self.fs_accs[i, 1, 1] += a[0]
            self.fs_accs[j, 1, 1] += a[1]
            self.fs_accs[k, 1, 1] += a[2]

        self.fs_accs /= total
        self.fs_accs[self.fs_accs > 1] = 0.9999

        # Flying Squid assumes symmetry, Pr(vote_i = 1 | y = 1) = Pr(vote_i = -1 | y = -1)
        self.fs_accs[:, 0, 0] = self.fs_accs[:, 1, 1]
        self.fs_accs[:, 1, 0] = 1 - self.fs_accs[:, 1, 1]
        self.fs_accs[:, 0, 1] = 1 - self.fs_accs[:, 1, 1]

    def _get_train_acc(self):
        """ Compute Pr(vote_i = y) on the training data.
        """
        self.train_acc = (self.train_votes.T == self.train_gold).mean(axis=1)
        self.train_no_val_acc = (self.train_no_val_votes.T == self.train_no_val_gold).mean(axis=1)



    def _get_test_acc(self):
        """ Compute Pr(vote_i = y) on the test data.
        """
        self.test_acc = (self.test_votes.T == self.test_gold).mean(axis=1)


    def pick_best(self): 
        """Use the predictor with the best performance on the train set.
        """
        self.best_prompt = np.argmax(self.train_acc)
        test_preds = self.test_votes[:, self.best_prompt]

        return accuracy_score(self.test_gold, test_preds)

    def majority_vote(self):
        """Take a majority vote over predictors. Current implementation ignores abstains.
        When there is a tie, we pick the prompt with the lowest index.
        When all prompts abstain, we just return the most common label argmax_y Pr(y).
        """ 

        test_preds = np.zeros(self.n_test)
        for i in range(self.n_test):
            # Majority vote discards abstains if any
            if self.abstains:
                voters = self.test_votes[i, self.test_votes[i] != self.abstain_value]
            else:
                voters = self.test_votes[i]

            counts = collections.Counter(voters)
            if len(counts) == 0:
                # e.g. all prompts abstain --> return most common class label
                test_preds[i] = self.balance.argmax()
            else:
                test_preds[i] = counts.most_common(1)[0][0]

        return test_preds.astype(int), accuracy_score(self.test_gold, test_preds)

    def get_clique_probs(self, idxs, vals, y, symmetric = False):
        """
            Computes marginal probability over votes indexed by idx, Pr(votes_idxs = vals | y), using training data.
        """

        if symmetric:
            truth_matrix = np.ones(self.n_train).astype(bool)
            agree = np.where(vals == y)[0]
            disagree = np.where((np.logical_and(vals != self.abstain_value, vals != y)) == True)[0]

            for i, lf in enumerate(idxs):
                if i in agree:
                    truth_matrix = np.logical_and(truth_matrix, self.train_votes[:, lf] == self.train_gold)
                elif i in disagree:
                    truth_matrix = np.logical_and(truth_matrix, self.train_votes[:, lf] != self.train_gold)
                else:
                    truth_matrix = np.logical_and(truth_matrix, self.train_votes[:, lf] == self.abstain_value)
        else:
            truth_matrix = np.ones(len(self.gold_idxs[y])).astype(bool)
            for i, lf in enumerate(idxs):
                truth_matrix = np.logical_and(truth_matrix, self.train_votes[self.gold_idxs[y], lf] == vals[i])


        if len(np.where(truth_matrix == True)[0]) == 0:
            return 0.00001

        if symmetric:
            return len(np.where(truth_matrix == True)[0]) / self.n_train
        else:
            return len(np.where(truth_matrix == True)[0]) / len(self.gold_idxs[y])


    def get_clique_probs_unlabeled(self, idxs, on_all_data=True):
        if on_all_data:
            votes = np.concatenate((self.train_votes_scaled, self.test_votes_scaled))
            n = self.n_train + self.n_test
        else:
            votes = self.test_votes_scaled
            n = self.n_test 


        l = len(idxs) 
        e_y = 2*self.balance[1] - 1
        vote_moment = votes[:, idxs].prod(axis=1).mean()
        if l % 2 == 0:
            # E[Y] * E[lfs] = E[lfs Y]
            acc = vote_moment * e_y
        else:
            acc = vote_moment / e_y




    def get_cond_probs(self, votes, y, accs, edgeset = None, symmetric=False, abstains_symmetric = True):
        """ Computes the probability Pr(votes, y) assuming conditional independence.

        Args:
        - votes: m element array of votes in {-1, 0, ..., k-1}
        - y: the value of the label, in {0, ..., k - 1}
        - accs: the accuracy array, e.g. directly learned from data or from FlyingSquid
        - edgeset: set of edges to factorize probability with
        - abstains_symmetric: do we assume Pr(vote_i = 0 | y) = Pr(vote_i = 0) or not?
        """
        pr_y = self.balance[y]
        prod = pr_y
        if edgeset is None:
            # in this case, do not need junction tree factorization. Just multiply accs together
            for p in range(len(votes)):
                if self.abstains and votes[p] == self.abstain_value:
                    if abstains_symmetric:
                    # we can drop Pr(lf_i = 0 | y) since it appears the same amount of times in numerator and denominator
                        prod *= (1 - self.coverage[p])
                        # print(f"multiplying by abstain on {p}: {1 - self.coverage[p]}")
                    else:
                        prod *= self.abstain_rate[p, y]
                else:
                    # print(f"multiplying by accuracy Pr(vote_{p} = {votes[p]} | y = {y}): {accs[p, y, votes[p]]}")
                    prod *= accs[p, y, votes[p]] # this assumes everything is independent
        
        else:
            # multiply over maximal cliques
            for i in self.c_tree.nodes():
                node = self.c_tree.nodes[i]
                members = list(node['members'])
                if len(members) == 1:
                    v = members[0]
                    if self.abstains and votes[v] == self.abstain_value:
                        if abstains_symmetric:
                            prod *= (1 - self.coverage[v])
                            # print(f"multiplying by abstain on {v}: {1 - self.coverage[v]}")
                        else:
                            #print("multiplying by abstains")
                            prod *= self.abstain_rate[v, y]
                    else:
                        #print(f"multiplying by accuracy of {v}: {accs[v, y, votes[v]] }")
                        prod *= accs[v, y, votes[v]] 
                        # print(f"multiplying by Pr(vote_{v} = {votes[v]} | y = {y}): {accs[v, y, votes[v]]}")

                else:
                    #print(f"multiplying by prob over clique {members}: {self.get_clique_probs(members, votes[members], y, symmetric)}")
                    prod *= self.get_clique_probs(members, votes[members], y, symmetric)

            # divide over separator sets      
            for i in self.c_tree.edges():
                edge = self.c_tree.edges[i]
                members = list(edge['members'])
                if len(members) == 1:
                    v = members[0]
                    deg = len(self.c_data[v]['max_cliques'])
                    if self.abstains and votes[v] == self.abstain_value:
                        if abstains_symmetric:
                            prod /= (1 - self.coverage[v])**(deg - 1)
                        else:
                            if self.abstain_rate[v, y] == 0:
                                prod /= 0.000001**(deg - 1) # edge cas
                            else:
                                prod /= self.abstain_rate[v, y]**(deg - 1)
                    else:
                        #print(f"Dividing by symmetric accuracy of {v}")
                        prod /= accs[v, y, votes[v]]**(deg - 1) 
                else:
                    #print(f"Dividing by prob over clique {members}: {self.get_clique_probs(members, votes[members], y, symmetric)}")
                    deg = len(self.c_data[tuple(members)]['max_cliques'])
                    prod /= (self.get_clique_probs(members, votes[members], y, symmetric))**(deg-1)
                            
        return prod 

    def get_probs(self, votes, accs, edgeset = None, symmetric=False, abstains_symmetric = True):
        """ Computes the probability Pr(y | votes) using Bayes Rule over Pr(votes, y).

        Args:
        - votes: m element array of votes in {-1, 0, ..., k-1}
        - accs: the accuracy array, e.g. directly learned from data or from FlyingSquid
        - edgeset: set of edges to factorize probability with
        - abstains_symmetric: do we assume Pr(vote_i = 0 | y) = Pr(vote_i = 0) or not?

        """
        p = np.zeros(self.k)
        for i in range(self.k):
            p[i] = self.get_cond_probs(votes, self.classes[i], accs, edgeset, symmetric, abstains_symmetric)

        

        p /= p.sum() # normalization
        return p

    def naive_bayes(self, accs = None, symmetric = False, abstains_symmetric=True):
        """ Naive bayes estimation.

        Estimate Pr(vote_i | y) from training data and use that to compute Pr(y = 1 | votes).
        Assumes conditional independence.

        Args:
        - accs: the accuracies [m x k x k] we estimate with
        - symmetric: Do we assume Pr(vote_i = c | y = c) = Pr(vote_i = y) for all c?
        - abstains_symmetric: Do we assume Pr(vote_i = 0 | y) = Pr(vote_i = 0)? This is 
        reasonable when an abstain is due to a systematic error in the prompt that doesn't depend on the label of the data.
        """        
        test_preds = []
        test_probs = []

        if symmetric:
            accs = self.sym_accs
        else:
            if accs is None:
                accs = self.nb_accs

        for votes in self.test_votes:
            prob = self.get_probs(votes, accs, symmetric=symmetric, abstains_symmetric=abstains_symmetric)
            test_probs.append(prob)
            test_preds.append(np.argmax(prob))

        return test_probs, accuracy_score(self.test_gold, test_preds)


    def junction_tree(self, edgeset, symmetric=False, abstains_symmetric=True, data='test'):
        """ Junction tree estimation.

        Estimate Pr(vote_i | y) from training data and use that to compute Pr(y = 1 | votes).
        Assumes edgeset structure.

        Args:
        - edgeset: List of tuples (i, j) for i, j in {0, ..., m} denoting edges to factorize distribution with.
        - symmetric: Do we assume Pr(vote_i = c | y = c) = Pr(vote_i = y) for all c?
        - abstains_symmetric: Do we assume Pr(vote_i = 0 | y) = Pr(vote_i = 0)? This is 
        reasonable when an abstain is due to a systematic error in the prompt that doesn't depend on the label of the data.

        """        

        # construct auxiliary data structures
        self._set_clique_tree(edgeset)
        self._set_clique_data()

        # Get preds 
        preds = []
        probs = []

        if data=='val':
            votes = self.val_votes 
            gold = self.val_gold
        elif data=='test':
            votes = self.test_votes 
            gold = self.test_gold 
        else:
            votes = self.train_votes 
            gold = self.train_gold 

        if symmetric:
            accs = self.sym_accs
        else:
            accs = self.nb_accs

        for v in votes:
            prob = self.get_probs(v, accs, edgeset, symmetric=False, abstains_symmetric= abstains_symmetric)
            probs.append(prob)
            preds.append(np.argmax(prob))

        return probs, accuracy_score(gold, preds)




    def conditional_entropy(self, votes, edgeset=None):
        """
            Computes H(Y | votes) ~= -1/n sum_i sum_y' Pr(y = y' | votes_j) log Pr(y = y' | votes_j). 
            Uses learned distribution as true one over Pr(y | votes)

            This computation is independent of aggregation approach. 
            It uses direct estimation on the training dataset to learn the PGM.
        """
        ce = 0

        if edgeset is not None:
            self._set_clique_tree(edgeset)
            self._set_clique_data()

        #print("Votes 1")
        #print(votes)

        for i, vote in enumerate(votes):
            # compute Pr(y | lf) for all y. We are treating this estimated probability as the true distribution.
            prob_vector = self.get_probs(vote, self.nb_test_accs, edgeset, symmetric=False, abstains_symmetric=True)

            # print(prob_vector, vote, i)
            
            # print(prob_vector, vote)
            for j in range(self.k):
                if prob_vector[j] == 0:
                    continue
                # print(vote, j, prob_vector[j])
                ce += prob_vector[j] * np.log(prob_vector[j])
        return -ce/len(votes)


    def conditional_entropy_singleton(self, probs, gold, edgeset=None):
        """
            Computes H(Y | WS output) = -1/n sum_i  sum_j Pr(y-hat = y_j | lfs(x_i)) * sum_k Pr(y = y_k | y-hat = y_j) log Pr(y = y_k | y-hat = y_j)
        """

        # First, compute WS estimate y-hat over dataset
        preds = np.argmax(probs, axis=1) # need to 


        # Now estimate Pr(y | y-hat) (k by k) matrix 
        y_accs = np.zeros((self.k, self.k))
        ws_idxs = [np.where(preds == c)[0] for c in self.classes]

        for i in range(self.k):
            for j in range(self.k):
                y_accs[i, j] = len(np.where(gold[ws_idxs[i]] == self.classes[j])[0]) / len(ws_idxs[i])

        # print(y_accs)

        # finally, compute entropy: 1/n sum_i  sum_j Pr(y-hat = y_j | lfs(x_i)) * sum_k Pr(y = y_k | y-hat = y_j) log Pr(y = y_k | y-hat = y_j)
        ce = 0
        for i in range(len(probs)):
            for j in range(self.k):
                for c in range(self.k):
                    y_prob = y_accs[c, j]
                    if y_prob == 0:
                        continue 
                    ce += probs[i, j] * y_prob * np.log(y_prob)

        return -ce/len(probs)

    def conditional_entropy_mv(self, edgeset=None):
        """
            Computes H(Y | MV output) = -1/n sum_i sum_k Pr(y = y_k | y-hat_i) log Pr(y = y_k | y-hat_i)

        """

        # First, compute MV estimate y-hat over dataset
        preds, _ = self.majority_vote()

        # Now estimate Pr(y | y-hat) (k by k) matrix 
        y_accs = np.zeros((self.k, self.k))
        ws_idxs = [np.where(preds == c)[0] for c in self.classes]
        for i in range(self.k):
            for j in range(self.k):
                y_accs[i, j] = len(np.where(self.test_gold[ws_idxs[i]] == self.classes[j])[0]) / len(ws_idxs[i])

        ce = 0
        for i, vote in enumerate(self.test_votes):
            v_pred = preds[i]
            for j in range(self.k):
                y_prob = y_accs[v_pred, j]
                if y_prob == 0:
                    continue 
                ce += y_prob * np.log(y_prob)

        return -ce/len(self.test_votes)


    def cross_entropy_conditional(self, votes, golds, edgeset):
        """
            Computes -1/n sum_i log Pr(y-hat = y | votes_j). This is the standard notion of CE loss.
        """
        ce = 0

        self._set_clique_tree(edgeset)
        self._set_clique_data()
        for i, vote in enumerate(votes):
            # compute Pr(y | lf)
            prob = self.get_probs(vote, self.nb_accs, edgeset, symmetric=False, abstains_symmetric=True)
            ce += np.log(prob[golds[i]])

        return -ce/len(votes)

    def cross_entropy(self, votes, golds, edgeset):
        """
            Computes -1/n sum_i log Pr(y-hat = y, votes_j), minimizing cross entropy over the joint distribution of Y, votes.
        """
        ce = 0

        self._set_clique_tree(edgeset)
        self._set_clique_data()
        for i, vote in enumerate(votes):
            # compute Pr(votes, y)
            prob = self.get_cond_probs(vote, golds[i], self.nb_accs, edgeset, symmetric=False, abstains_symmetric=True)
            ce += np.log(prob)

        return -ce/len(votes)

    def cross_entropy_no_label(self, votes, edgeset):
        """
            Computes -1/n sum_j log Pr(votes_j), minimizing cross entropy over the distribution of votes
        """
        ce = 0
        self._set_clique_tree(edgeset)
        self._set_clique_data()

        for i, vote in enumerate(votes):
            # compute Pr(votes, y)
            prob = 0
            for c in self.classes:
                prob+= self.get_cond_probs(vote, c, self.nb_accs, edgeset, symmetric=False, abstains_symmetric=True)
            ce += np.log(prob)

        return -ce/len(votes)


    def flying_squid(self, abstains_symmetric=True):
        """ FlyingSquid algorithm requires no labeled data (except for estimating class balance).

        Assumes conditional independence (for now) and symmetric accuracies. 
        That is, Pr(vote_i = 1 | y = 1) = Pr(vote_i = 0 | y = 0).

        Args:
        - abstains_symmetric: Do we assume Pr(vote_i = 0 | y) = Pr(vote_i = 0)? This is 
        reasonable when an abstain is due to a systematic error in the prompt that doesn't depend on the label of the data.
        """
        assert self.k == 2, "not implemented for more than 2 classes!"
        assert 0 in self.classes 
        assert 1 in self.classes

        # Get preds 
        test_preds = []
        test_probs = []
        for votes in self.test_votes:
            prob = self.get_probs(votes, self.fs_accs, abstains_symmetric=abstains_symmetric)
            test_probs.append(prob)
            test_preds.append(np.argmax(prob))

        return test_probs, accuracy_score(self.test_gold, test_preds)


    def snorkel_lm(self, on_all_data=True):
        """ Use Snorkel AI's label model. Under the hood: Metal "forward" algorithm.
        """
        #assert self.k == 2, "not implemented for more than 2 classes!"
        #assert 0 in self.classes 
        #assert 1 in self.classes

        if on_all_data:
            votes = np.concatenate((self.train_votes, self.test_votes))
            n = self.n_train + self.n_test
        else:
            votes = self.test_votes
            n = self.n_test 


        label_model = LabelModel(cardinality=self.k)

        label_model.fit(L_train=votes, n_epochs=500, log_freq=100, seed=0)
        probs_test = label_model.predict_proba(self.test_votes)
        test_preds = np.argmax(probs_test, axis=1)
        params = label_model.get_conditional_probs()

        return params, accuracy_score(self.test_gold, test_preds)


    def dp_learn_params(self, with_label=False, seed=0, lr=0.0001, epochs = 1000):
        """ Learn the data programming parameters alpha and beta.

        Args:
        - with_label: Do we use y or not? If using label, use the train set and do MLE on Pr(y, votes); 
        else use the test set and do MLE on Pr(votes).
        - seed: random seed for Pytorch.
        - lr: learning rate
        - epochs: number of epochs

        Returns:
        - alpha: parameter corresponding to accuracy Pr(vote_i = y) (symmetric)!
        - beta: parameter corresponding to coverage Pr(vote_i != 0) (symmetric)!
        """
        if with_label:
            votes = self.train_votes_scaled
            gold = self.train_gold_scaled
        else:
            votes = self.test_votes_scaled
            gold = self.test_gold_scaled

        torch.manual_seed(seed)

        x = torch.from_numpy(votes).type(torch.FloatTensor)

        # Initialize Parameters
        alpha = torch.rand(self.m, requires_grad=True)
        beta = torch.tensor(self.coverage, requires_grad = False).type(torch.FloatTensor) # we do not optimize beta for now
        optimizer = torch.optim.SGD([alpha], lr=lr)

        for t in range(epochs):
            optimizer.zero_grad()

            mu_1 = torch.prod((x == 1) * beta.multiply(alpha) + (x == -1) * beta.multiply(1 - alpha) + (x == 0) * (1 - beta), dim = 1)
            mu_neg1 = torch.prod((x == -1) * beta.multiply(alpha) + (x == 1) * beta.multiply(1 - alpha) + (x == 0) * (1 - beta), dim=1)

            if with_label:
                # use the label information in MLE 
                snorkel_loss = -torch.log(mu_1[np.where(gold == 1)[0]]).sum() - torch.log(mu_neg1[np.where(gold == -1)[0]]).sum()
            else:
                # 50-50 for y = 1 vs -1
                snorkel_loss = -torch.log(0.5*mu_1 + 0.5*mu_neg1).sum()
    
            snorkel_loss.backward()
            optimizer.step()

            #if t % 100 == 0:
            # print('Loss', snorkel_loss, 'alpha', alpha, 'beta', beta)


            with torch.no_grad():
                alpha.clamp_(0.5, 1) # assume that accuracy is better than random
                beta.clamp_(0, 1) # coverage should be between 0 and 1
                
        return alpha, beta

    def data_programming(self, with_label=False, seed=0, lr = 0.0001, epochs=1000):
        """ Data programming algorithm.
        Args:
        - with_label: Do we use y or not? If using label, use the train set and do MLE on Pr(y, votes); 
        else use the test set and do MLE on Pr(votes).
        - seed: random seed for Pytorch.
        - lr: learning rate
        - epochs: number of epochs
        """

        assert self.k == 2, "not implemented for more than 2 classes!"
        assert 0 in self.classes 
        assert 1 in self.classes

        # we don't need betas, will just cancel out when doing inference
        alpha, beta = self.dp_learn_params(with_label, seed, lr, epochs)
        alpha = alpha.detach().numpy()

        if np.any(np.isnan(alpha)):
            raise ValueError("SGD failed to converge.")

        dp_accs = np.zeros((self.m, 2, 2))
        dp_accs[:, 1, 1] = dp_accs[:, 0, 0] = alpha 
        dp_accs[:, 1, 0] = dp_accs[:, 0, 1] = 1 - alpha

        if with_label:
            self.dp_label_accs = dp_accs 
        else:
            self.dp_nolabel_accs = dp_accs

        # Get preds 
        test_preds = []
        test_probs = []
        for votes in self.test_votes:
            prob = self.get_probs(votes, dp_accs)
            test_probs.append(prob[1])
            test_preds.append(np.argmax(prob))

        return test_probs, accuracy_score(self.test_gold, test_preds)



    def logistic_regression(self, pairwise=True, singleton=False, scaling=True, max_iter=100):
        """ 
        Logistic regression baseline.

        Args:
        - pairwise: if true, we scale everything to [-1, 1] and look at vote_i * vote_j as (m choose 2) new features that 
        explicitly model their agreement and disagreement.
        - singleton: do we include the original votes as features
        - scaling: do logistic regression over [-1, 1] or [0, 1]
        - max_iter: maximum number of iterations for sklearn LR algorithm.
        """
        if scaling:
            train_votes = self.train_no_val_votes_scaled 
            val_votes = self.val_votes_scaled
            test_votes = self.test_votes_scaled

            train_gold = self.train_no_val_gold_scaled
            val_gold = self.val_gold_scaled
            test_gold = self.test_gold_scaled
        else: 
            train_votes = self.train_no_val_votes
            val_votes = self.val_votes
            test_votes = self.test_votes

            train_gold = self.train_no_val_gold
            val_gold = self.val_gold
            test_gold = self.test_gold

        if pairwise:
            # get pairwise data
            pair_idxs = list(itertools.combinations(np.arange(self.m), 2))

            pairs_train = np.zeros((len(train_gold), len(pair_idxs)))
            pairs_val = np.zeros((len(val_gold), len(pair_idxs)))
            pairs_test = np.zeros((self.n_test, len(pair_idxs)))

            for count, (i, j) in enumerate(pair_idxs):
                pairs_train[:, count] = train_votes[:, i] * train_votes[:, j]
                pairs_val[:, count] = val_votes[:, i] * val_votes[:, j]
                pairs_test[:, count] = test_votes[:, i] * test_votes[:, j]


            if not singleton:
                train_votes = pairs_train
                val_votes = pairs_val
                test_votes = pairs_test 

            else:
                train_votes = np.concatenate((train_votes, pairs_train), axis=1)
                val_votes = np.concatenate((val_votes, pairs_val), axis=1)
                test_votes = np.concatenate((test_votes, pairs_test), axis=1)


        best_val = -1
        best_test = -1
        best_reg = -1

        # grid search over regularization parameter using validation set
        for c in [0.001, 0.01, 0.1, 0.25, 0.5, 5, 10, 100, 1000, 2000]:
            clf = LogisticRegression(random_state=0, penalty='l1', solver='liblinear', fit_intercept=False, multi_class='ovr', C=c, max_iter=max_iter).fit(train_votes, train_gold)
            test_score = clf.score(test_votes, test_gold)
            val_score = clf.score(val_votes, val_gold)

            if val_score >= best_val:
                best_val = val_score
                best_test = test_score
                best_reg = c

        clf = LogisticRegression(random_state=0, penalty='l1', solver='liblinear', fit_intercept=False, multi_class='ovr', C=best_reg, max_iter=max_iter).fit(train_votes, train_gold)
        return clf.coef_, clf.predict_proba(test_votes), best_test


    def exp_weight(self, option=1, etas=[0.25, 0.5, 1, 2, 4, 8, 16, 32]):
        """
        Weighting rule 1: Pr(y | votes) ~ sum_i 1{vote_i = y} * exp(-eta*loss_i)
        Weighting rule 2: Pr(y | votes) ~ exp(sum_i eta * accuracy * vote_i * y) (scaled to -1, 1)

        Args:
        - option: which weighting rule to use
        - etas: list of temperature hyperparameters
        """
        test_preds = []
        test_probs = []

        # grid search
        best_eta = -1
        best_acc = 0
        for eta in etas:
            val_preds = []
            if option == 1:
                weights = np.exp(-eta * (1 - self.train_no_val_acc))
            else:
                weights = eta*self.train_no_val_acc
            for votes in self.val_votes:
                if option == 1:
                    scores = np.array([weights[votes == y].sum() for y in self.classes])
                else:
                    scores = np.array([np.exp((2*((votes == y).astype(int))-1).dot(weights)) for y in self.classes] )
                if scores.sum() ==0:
                    # return prior
                    val_preds.append(np.argmax(self.balance))
                else:
                    val_preds.append(np.argmax(scores))

            val_acc = accuracy_score(self.val_gold, val_preds)
            if val_acc > best_acc:
                best_eta = eta

        if option == 1:
            weights = np.exp(-best_eta * (1 - self.train_no_val_acc))
        else:
            weights = best_eta*self.train_no_val_acc
        
        for votes in self.test_votes:
            if option == 1:
                scores = np.array([weights[votes == y].sum() for y in self.classes])
            else:
                scores = np.array([np.exp((2*((votes == y).astype(int))-1).dot(weights)) for y in self.classes] )
            if scores.sum() ==0:
                # return prior
                test_preds.append(np.argmax(self.balance))
            else:
                scores /= scores.sum()

                test_probs.append(scores[1])
                test_preds.append(np.argmax(scores))

        return test_probs, accuracy_score(self.test_gold, test_preds)


class MultiAggregator(Aggregator):

    def __init__(self, train_votes, train_gold, test_votes, test_gold, classes, abstains=False, abstain_value=-1) -> None:
        super().__init__(train_votes, train_gold, test_votes, test_gold, abstains, classes, abstain_value)


    def flying_squid(self, abstains_symmetric=True):
        """
        For multi-class, FlyingSquid reduces into one-vs-all subproblems and picking the highest Pr(y | votes) from each of those.
        """
        probs = np.zeros((self.n_test, self.k))
        for i, c in enumerate(self.classes):

            train_votes_c = np.where(self.train_votes == c, 1, 0)
            train_votes_c[self.train_votes == self.abstain_value] = -1 # keep the abstains

            train_gold_c = np.where(self.train_gold == c, 1, 0)

            test_votes_c = np.where(self.test_votes == c, 1, 0)
            test_votes_c[self.test_votes == self.abstain_value] = -1

            test_gold_c = np.where(self.test_gold == c, 1, 0)

            agg =  Aggregator(train_votes_c, train_gold_c, test_votes_c, test_gold_c, self.abstains, classes=[0, 1])

            fs_probs, fs_acc = agg.flying_squid(abstains_symmetric)
            probs[:, i] = np.array(fs_probs)[:, 1]


        test_preds = np.argmax(probs, axis=1)
        return probs, accuracy_score(self.test_gold, test_preds)



    def data_programming(self, epochs=1000, with_label=False, seed=0, lr=0.0001):
        """
        For multi-class, data programming reduces into one-vs-all subproblems and picking the highest Pr(y | votes) from each of those.
        """

        probs = np.zeros((self.n_test, self.k))
        # one versus all
        for i, c in enumerate(self.classes):
            train_votes_c = np.where(self.train_votes == c, 1, 0)
            train_votes_c[self.train_votes == self.abstain_value] = -1

            train_gold_c = np.where(self.train_gold == c, 1, 0)

            test_votes_c = np.where(self.test_votes == c, 1, 0)
            test_votes_c[self.test_votes == self.abstain_value] = -1

            test_gold_c = np.where(self.test_gold == c, 1, 0)

            agg =  Aggregator(train_votes_c, train_gold_c, test_votes_c, test_gold_c, self.abstains, classes=[0, 1])

            probs[:, i], _ = agg.data_programming(with_label, seed, lr, epochs)

        test_preds = np.argmax(probs, axis=1)
        return accuracy_score(self.test_gold, test_preds)

        


