import numpy as np
import itertools
import scipy.stats
import math
import networkx as nx

from itertools import chain

from methods import Aggregator 
from binary_deps import structure_learning
from binary_deps import DependentPGM

from sklearn.metrics import log_loss, accuracy_score

class Ising():
    
    
    def __init__(self, m, potentials, thetas = None, vals = [-1, 1], ) -> None:
        self.m = m 
        self.v = m + 1 # total number of vertices 
        self.potentials = potentials 

        self.vals = vals
        #TODO support values in 0, 1

        if thetas is not None:
            assert len(thetas) >= len(potentials), f"Need to specify at least {len(potentials)} theta parameters."
            self.thetas = thetas 
        else:
            self.thetas = np.random.rand(len(potentials))

        # 2^v size support over y and all prompts
        self.support = np.array(list(map(list, itertools.product(vals, repeat=self.v))))

        # 2^m size support over all prompts
        self.support_no_y = np.array(list(map(list, itertools.product(vals, repeat=self.m))))

        self.n_vals = len(self.support)

        self._make_pdf()
        self._make_cdf()

        self._get_means()
        self._get_balance()
        self._get_accs()

        # set graph true graph structure
        self._get_edges_nodes()
        self.c_tree = self._set_clique_tree(self.edges)
        self.c_data = self._set_clique_data(self.c_tree)


    def _get_edges_nodes(self):
        self.nodes = np.arange(self.m)
        self.edges = [p for p in self.potentials if len(p) == 2 and self.m not in p]
        if self.edges != []:
            self.higher_order = True
        else:
            self.higher_order = False

    def _exponential_family(self, labels):
        x = 0.0
        for i in range(len(self.potentials)):
            x += self.thetas[i] * labels[self.potentials[i]].prod()

        return np.exp(x)

    def _make_pdf(self):
        p = np.zeros(len(self.support))
        for i, labels in enumerate(self.support):
            p[i] = self._exponential_family(labels)

        self.z = sum(p)
        
        self.pdf = p/self.z

    def _make_cdf(self):
        self.cdf = np.cumsum(self.pdf)

    def joint_p(self, C, values):
        p = 0.0
        for k, labels in enumerate(self.support):
            flag = True 
            for i in range(len(C)):
                prod = labels[C[i]].prod() 
                if prod != values[i]:
                    flag = False 

            if flag == True:
                p += self.pdf[k]

        return p

    def expectation(self, C):
        return self.vals[0] * self.joint_p(C, self.vals[0] * np.ones(len(C))) + self.vals[1] * self.joint_p(C, self.vals[1] * np.ones(len(C)))

    def _get_means(self):
        self.means = np.zeros(self.m)
        for k in range(self.m):
            self.means[k] = self.expectation([[k]])


    def _get_balance(self):
        self.balance = self.joint_p([[self.m]], [1])

    def _get_covariance_y(self):
        self.cov = np.zeros((self.m + 1, self.m + 1))

    def aug_covariance(self, rvs):
        l = len(rvs)
        M = np.zeros((l, l))
        for i in range(l):
            for j in range(i + 1, l):
                M[i, j] = self.joint_p([rvs[i], rvs[j]], [1, 1]) + self.joint_p([rvs[i], rvs[j]], [-1, -1])
        
        for i in range(l):
            for j in range(i + 1):
                if i != j: 
                    M[i, j] = M[j, i]
                else:
                    M[i, j] = 1
        
        M = 2*M - 1
        
        mu = np.zeros(l)
        for i in range(l):
            mu[i] = self.joint_p([rvs[i]], [1])
        mu = 2*mu - 1
        
        return M - np.outer(mu, mu)

    def aug_covariance_y(self, rvs, y):
        p_y = self.balance if y == 1 else 1 - self.balance
        l = len(rvs)
        M = np.zeros((l, l))
        for i in range(l):
            for j in range(i + 1, l):
                M[i, j] = (self.joint_p([rvs[i], rvs[j], [self.m]], [1, 1, y]) + self.joint_p([rvs[i], rvs[j], [self.m]], [-1, -1, y])) / p_y
        
        for i in range(l):
            for j in range(i + 1):
                if i != j: 
                    M[i, j] = M[j, i]
                else:
                    M[i, j] = 1
        

        M = 2*M - 1
        
        mu = np.zeros(l)
        for i in range(l):
            mu[i] = self.joint_p([rvs[i], [self.m]], [1, y]) / p_y
        mu = 2*mu - 1
        
        return M - np.outer(mu, mu)


    def _get_accs(self):
        """
            self.accs[k, i, j] = Pr(lf_k = j | y = i) (i, j scaled to -1, 1 if needed)
        """
        self.accs = np.zeros((self.m, 2, 2))
        for k in range(self.m):
            self.accs[k, 1, 1] = self.joint_p([[k], [self.m]], [self.vals[1], self.vals[1]]) / self.balance 
            self.accs[k, 0, 0] = self.joint_p([[k], [self.m]], [self.vals[0], self.vals[0]]) / (1 - self.balance)
            self.accs[k, 1, 0] = 1 - self.accs[k, 1, 1]
            self.accs[k, 0, 1] = 1 - self.accs[k, 0, 0]


    def sample(self):
        r = np.random.random_sample() 
        smaller = np.where(self.cdf < r)[0]
        if len(smaller) == 0:
            i = 0 
        else: 
            i = smaller.max() + 1

        return self.support[i]

    def make_data(self, n, has_label = True):
        L = np.zeros((n, self.m))
        gold = np.zeros(n)
        for i in range(n):
            l = self.sample()
            L[i, :] = l[:self.m]

            if has_label:
                gold[i] = l[self.m]

        return L.astype(int), gold.astype(int)



    def _set_clique_tree(self, edges):
        G1 = nx.Graph()
        G1.add_nodes_from(self.nodes)
        G1.add_edges_from(edges)
        
        # Check if graph is chordal
        # TODO: Add step to triangulate graph if not
        if not nx.is_chordal(G1):
            raise NotImplementedError("Graph triangulation not implemented.")

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

        return nx.maximum_spanning_tree(G2) # should be maximum??? Because we want maximal separator sets
        # Return a minimum spanning tree of G2

    def _set_clique_data(self, c_tree):
        # Create a helper data structure which maps cliques (as tuples of member
        # sources) --> {start_index, end_index, maximal_cliques}, where
        # the last value is a set of indices in this data structure    
        c_data = dict()
        for i in range(self.m):
            c_data[i] = {
                "vertices": [i],
                "max_cliques": set( # which max clique i belongs to
                    [
                        j
                        for j in c_tree.nodes()
                        if i in c_tree.nodes[j]["members"]
                    ]
                ),
            }

        # Get the higher-order clique statistics based on the clique tree
        # First, iterate over the maximal cliques (nodes of c_tree) and
        # separator sets (edges of c_tree)
        if self.higher_order:
            counter = 0
            for item in chain(c_tree.nodes(), c_tree.edges()):
                if isinstance(item, int):
                    C = c_tree.nodes[item]
                    C_type = "node"
                elif isinstance(item, tuple):
                    C = c_tree[item[0]][item[1]]
                    C_type = "edge"
                else:
                    raise ValueError(item)
                members = list(C["members"])
                nc = len(members)

                # Else add one column for each possible value
                if nc != 1:
                    # Add to self.c_data as well
                    #idx = counter + m
                    c_data[tuple(members)] = {
                        "vertices": members,
                        "max_cliques": set([item]) if C_type == "node" else set(item),
                    }
                    counter += 1
        return c_data


    def get_cond_probs(self, votes, y, edgeset = None):
        """
            Computes the probability Pr(votes | y).
        """
        pr_y = self.balance if y == 1 else 1 - self.balance
        prod = pr_y

        votes_scaled = 2*votes - 1
        y_scaled = 2*y - 1


        if edgeset is not None:
            c_tree = self._set_clique_tree(edgeset)
            c_data = self._set_clique_data(c_tree)
        else:
            c_tree = self.c_tree 
            c_data = self.c_data
        
        for i in c_tree.nodes():
            node = c_tree.nodes[i]
            members = list(node['members'])
            if len(members) == 1:
                v = members[0]
                prod *= self.accs[v, y, votes[v]]
            else:
                # prod *= self.get_clique_probs(members, votes[members], y)
                member_votes = np.append(votes_scaled[members], y_scaled)
                members = [[m] for m in members] + [[self.m]]
                clique_probs = self.joint_p(members, member_votes)/self.joint_p([[self.m]], [y_scaled])
                #print("clique probs")
                #print(members, member_votes)
                #print(self.joint_p(members, member_votes))
                #print(clique_probs)
                prod *= clique_probs
                
        for i in c_tree.edges():
            edge = c_tree.edges[i]
            members = list(edge['members'])
            if len(members) == 1:
                v = members[0]
                deg = len(c_data[v]['max_cliques'])
                prod /= (self.accs[v, y, votes[v]])**(deg-1)
            else:
                deg = len(c_data[tuple(members)]['max_cliques'])
                # prod /= (self.get_clique_probs(members, votes[members], y))**(deg-1)

                member_votes = np.concatenate(votes[members], y_scaled)
                members = [[m] for m in members] + [[self.m]]
                clique_probs = self.joint_p(members, member_votes)/self.joint_p([[self.m]], [y_scaled])
                prod /= clique_probs**(deg-1)


        return prod 


    def get_probs(self, votes, edgeset = None):
        """
            Computes the probability Pr(y = 1 | votes).
        """
        pos = self.get_cond_probs(votes, 1, edgeset)
        neg = self.get_cond_probs(votes, 0, edgeset)
        if pos == 0:
            return 0
        else:
            return pos / (pos + neg)



    def cross_entropy(self, edgeset):
        ce = 0
        for i in range(self.n_vals):
            votes_unscaled = (0.5*(self.support[i, :self.m]+1)).astype(int)
            y_unscaled = int(0.5*(self.support[i, self.m]+1))
            ce += self.pdf[i] * np.log(self.get_cond_probs(votes_unscaled, y_unscaled, edgeset))
        return -ce
        
    def cross_entropy_conditional(self, edgeset):
        ce = 0
        for i in range(self.n_vals):
            votes_unscaled = (0.5*(self.support[i, :self.m]+1)).astype(int)
            y_unscaled = int(0.5*(self.support[i, self.m]+1))

            prob = self.get_probs(votes_unscaled, edgeset)
            if y_unscaled == 0:
                prob = 1 - prob
            ce += self.pdf[i] * np.log(prob)
        return -ce


    def cross_entropy_no_label(self, edgeset):
        ce = 0
        for i in range(len(self.support_no_y)):
            sequence = self.support_no_y[i]
            sequence_scaled = (0.5*(sequence+1)).astype(int) # scale back to 0/1

            voters = [[i] for i in np.arange(self.m)]

            true_prob = self.joint_p(voters, sequence) 

            pos = self.get_cond_probs(sequence_scaled, 1, edgeset)
            neg = self.get_cond_probs(sequence_scaled, 0, edgeset)

            ce += true_prob * np.log(pos + neg)
        return -ce


def to01(labels):
    return (0.5*(labels + 1)).astype(int)

# MV and picking the best prompt should do well when things are conditionally independent and equally same
def test0():
    m = 3

    thetas = [0, 0.5, 0.5, 0.5]
    # all conditionally independent, some singletons are fine 
    potentials = [[3], [0, 3], [1, 3], [2, 3]]

    pgm = Ising(m, potentials, thetas)

    # make data
    n_train = 1000
    train_votes, train_gold = pgm.make_data(n_train)
    n_test = 1000
    test_votes, test_gold = pgm.make_data(n_test)

    train_votes = to01(train_votes)
    train_gold = to01(train_gold)
    test_votes = to01(test_votes)
    test_gold = to01(test_gold)

    agg = Aggregator(train_votes, train_gold, test_votes, test_gold)

    _, mv_acc = agg.majority_vote()
    pb_acc = agg.pick_best()
    _, nb_acc = agg.naive_bayes()
    _, sym_acc = agg.naive_bayes(symmetric=True)

    
    print(f"Majority vote: {mv_acc}")
    print(f"Pick the best: {pb_acc}")
    print(f"Naive bayes: {nb_acc}")
    print(f"Naive bayes (symmetric): {sym_acc}") # should be worse! 


    print(f"Test passed: {nb_acc == mv_acc}\n ")


def test1():
    m = 5

    # randomly parametrize exponential family to determine accuracies and correlations
    np.random.seed(2)
    thetas = np.random.rand(30)

    # all conditionally independent, some singletons are fine 
    potentials = [[5], [0], [1], [4], [0, 5], [1, 5], [2, 5], [3, 5], [4, 5]]

    pgm = Ising(m, potentials, thetas)

    # make data
    n_train = 1000
    train_votes, train_gold = pgm.make_data(n_train)
    n_test = 1000
    test_votes, test_gold = pgm.make_data(n_test)

    train_votes = to01(train_votes)
    train_gold = to01(train_gold)
    test_votes = to01(test_votes)
    test_gold = to01(test_gold)

    agg = Aggregator(train_votes, train_gold, test_votes, test_gold)

    _, mv_acc = agg.majority_vote()
    pb_acc = agg.pick_best()
    _, nb_acc = agg.naive_bayes()
    _, sym_acc = agg.naive_bayes(symmetric=True)

    print(f"Majority vote: {mv_acc}")
    print(f"Pick the best: {pb_acc}")
    print(f"Naive bayes: {nb_acc}")
    print(f"Naive bayes (symmetric): {sym_acc}") # should be worse! 

    print(f"Test passed: {nb_acc >= max(mv_acc, pb_acc) and sym_acc < nb_acc}\n ")
    

def test2():
    m = 3

    # randomly parametrize exponential family to determine accuracies and correlations
    #theta = np.random.rand()
    #theta_cliques = (np.random.randint(0, 2, 5)*2 - 1)*theta
    #theta = np.random.rand()
    #theta_cliques = [1, 1, 1, 1, 1, 1, 1]

    np.random.seed(3)
    thetas = np.random.rand(30)

    # all conditionally independent
    potentials = [[3], [0, 3], [1, 3], [2, 3]]

    pgm = Ising(m, potentials, thetas)

    n_train = 100
    train_votes, train_gold = pgm.make_data(n_train)
    n_test = 100
    test_votes, test_gold = pgm.make_data(n_test)

    train_votes = to01(train_votes)
    train_gold = to01(train_gold)
    test_votes = to01(test_votes)
    test_gold = to01(test_gold)
    

    agg = Aggregator(train_votes, train_gold, test_votes, test_gold)

    _, mv_acc = agg.majority_vote()
    pb_acc = agg.pick_best()
    _, nb_acc = agg.naive_bayes()
    _, sym_acc = agg.naive_bayes(symmetric=True)
    _, fs_acc = agg.flying_squid()


    print(pgm.joint_p([[3]], [1]))
    print(pgm.balance)
    print(pgm.expectation([[3]]), pgm.expectation([[0]]), pgm.expectation([[0, 3]]))
    print(pgm.expectation([[3]])*pgm.expectation([[0, 3]]))


    print(f"Majority vote: {mv_acc}")
    print(f"Pick the best: {pb_acc}")
    print(f"Naive bayes: {nb_acc}")
    print(f"Naive bayes (symmetric): {sym_acc}")
    print(f"FlyingSquid: {fs_acc}")


    print(f"Test passed: {fs_acc >= max(mv_acc, pb_acc) and nb_acc == sym_acc}\n")

def test3():
    m = 3

    np.random.seed(2)

    thetas = [0.5, 0.1, 0.4, 0.4]

    # all conditionally independent
    potentials = [[3], [0, 3], [1, 3], [2, 3]]

    pgm = Ising(m, potentials, thetas)

    # print(pgm.balance)

    n_train = 1000
    train_votes, train_gold = pgm.make_data(n_train)
    n_test = 1000
    test_votes, test_gold = pgm.make_data(n_test)


    train_votes = to01(train_votes)
    train_gold = to01(train_gold)
    test_votes = to01(test_votes)
    test_gold = to01(test_gold)

    agg = Aggregator(train_votes, train_gold, test_votes, test_gold)

    #print(agg.fs_accs, agg.nb_accs, pgm.accs)

    _, mv_acc = agg.majority_vote()
    pb_acc = agg.pick_best()
    _, nb_acc = agg.naive_bayes()
    _, fs_acc = agg.flying_squid()
    _, dp_nolabel_acc = agg.data_programming(with_label=False)
    _, dp_label_acc = agg.data_programming(with_label=True)


    #print(agg.dp_learn_params(with_label=False))

    print(f"Majority vote: {mv_acc}")
    print(f"Pick the best: {pb_acc}")
    print(f"Naive bayes: {nb_acc}")
    print(f"FlyingSquid: {fs_acc}")
    print(f"Data Programming (no label): {dp_nolabel_acc}")
    print(f"Data Programming (with label): {dp_label_acc}")

    assert 0.69 <= mv_acc <= 0.7 and 0.69 <= pb_acc <= 0.7, f"MV and pick best should be 0.692 and 0.694."
    assert 0.77 <= min(nb_acc, fs_acc, dp_nolabel_acc, dp_label_acc) <= 0.79, f"All methods should have accuracy 0.78."

    print(f"Test passed: {min(nb_acc, fs_acc, dp_nolabel_acc, dp_label_acc) >= max(mv_acc, pb_acc)}\n")


def test4():
    m = 3

    # randomly parametrize exponential family to determine accuracies and correlations
    #theta = np.random.rand()
    #theta_cliques = (np.random.randint(0, 2, 5)*2 - 1)*theta
    #theta = np.random.rand()
    #theta_cliques = [1, 1, 1, 1, 1, 1, 1]

    np.random.seed(3)
    thetas = np.random.rand(30)
    thetas[0] = 0.1
    thetas[1] = 0.2
    thetas[2] = 0.01
    thetas[3] = 0.1
    thetas[4] = 0.5 # make this hugeeee

    potentials = [[3], [0, 3], [1, 3], [2, 3], [0, 1]]

    pgm = Ising(m, potentials, thetas)

    print(pgm.joint_p([[0], [1], [3]], [1, 1, 1])/pgm.balance)
    print(pgm.joint_p([[0], [3]], [1, 1]) * pgm.joint_p([[1], [3]], [1, 1]) / pgm.balance**2)


    print(pgm.joint_p([[0], [2], [3]], [1, 1, 1])/pgm.balance)
    print(pgm.joint_p([[0], [3]], [1, 1]) * pgm.joint_p([[2], [3]], [1, 1]) / pgm.balance**2)

    n_train = 10000
    train_votes, train_gold = pgm.make_data(n_train)
    n_test = 1000
    test_votes, test_gold = pgm.make_data(n_test)

    train_votes = to01(train_votes)
    train_gold = to01(train_gold)
    test_votes = to01(test_votes)
    test_gold = to01(test_gold)

    print(pgm.expectation([[3]]), pgm.expectation([[0, 1]]), pgm.expectation([[0, 1, 3]]))
    print(pgm.expectation([[3]])*pgm.expectation([[0, 1]]))


    edgeset = [(0, 1)]

    agg = Aggregator(train_votes, train_gold, test_votes, test_gold)

    _, mv_acc = agg.majority_vote()
    pb_acc = agg.pick_best()
    nb_probs, nb_acc = agg.naive_bayes()
    fs_probs, fs_acc = agg.flying_squid()
    jt_probs, jt_acc = agg.junction_tree(edgeset)
    jt_sym_probs, jt_sym_acc = agg.junction_tree(edgeset, symmetric=True)

    # print(agg.fs_accs, agg.nb_accs, pgm.accs)

    #print(pgm.joint_p([[0], [1], [3]], [1, 1, 1]) / pgm.balance)
    #print(pgm.joint_p([[0], [1], [3]], [-1, -1, -1]) / pgm.balance)


    print(f"Majority vote: {mv_acc}")
    print(f"Pick the best: {pb_acc}")
    print(f"Naive bayes: {nb_acc}")
    print(f"FlyingSquid: {fs_acc}")
    print(f"Junction tree (with deps): {jt_acc}")
    print(f"Junction tree (with deps, symmetric): {jt_sym_acc}")

    print(agg.get_probs(np.array([1, 1, 0]), agg.sym_accs, edgeset=[(0, 1)], symmetric=True, abstains_symmetric=False))
    print(agg.get_probs(np.array([1, 1, 0]), agg.nb_accs, edgeset=[(0, 1)], symmetric=False, abstains_symmetric=False))
    print(pgm.get_probs(np.array([1, 1, 0])))



    fail = False
    for i, votes in enumerate(test_votes):
        if np.abs(pgm.get_probs(votes) - jt_probs[i][1]) > 0.05:
            print(votes)
            print(pgm.get_probs(votes), jt_probs[i][1])
            fail = True  

        #print(pgm.get_probs(votes), nb_probs[i], test_gold[i])
        #print(np.round(pgm.get_probs(votes)), np.round(fs_probs[i]), test_gold[i])


    if fail:
        print("Test failed.")
    else:
        print("Test passed.")

def test5():
    m = 3

    # randomly parametrize exponential family to determine accuracies and correlations
    #theta = np.random.rand()
    #theta_cliques = (np.random.randint(0, 2, 5)*2 - 1)*theta
    #theta = np.random.rand()
    #theta_cliques = [1, 1, 1, 1, 1, 1, 1]

    np.random.seed(3) # 6 is good , 9 , 10
    thetas = np.random.rand(30)
    thetas[0] = 0
    thetas[1] = 0.1
    thetas[2] = 0.6
    thetas[3] = 0.1
    thetas[4] = 0.6 # make this hugeeee

    potentials = [[3], [0, 3], [1, 3], [2, 3], [0, 1]]

    pgm = Ising(m, potentials, thetas)

    print(pgm.joint_p([[0], [1], [3]], [1, 1, 1])/pgm.balance)
    print(pgm.joint_p([[0], [3]], [1, 1]) * pgm.joint_p([[1], [3]], [1, 1]) / pgm.balance**2)


    print(pgm.joint_p([[0], [2], [3]], [1, 1, 1])/pgm.balance)
    print(pgm.joint_p([[0], [3]], [1, 1]) * pgm.joint_p([[2], [3]], [1, 1]) / pgm.balance**2)

    n_train = 10000
    train_votes, train_gold = pgm.make_data(n_train)
    n_test = 100
    test_votes, test_gold = pgm.make_data(n_test)

    train_votes = to01(train_votes)
    train_gold = to01(train_gold)
    test_votes = to01(test_votes)
    test_gold = to01(test_gold)

    edgeset = [(0, 1)]

    agg = Aggregator(train_votes, train_gold, test_votes, test_gold)

    _, mv_acc = agg.majority_vote()
    pb_acc = agg.pick_best()
    nb_probs, nb_acc = agg.naive_bayes()
    fs_probs, fs_acc = agg.flying_squid()
    jt_probs, jt_acc = agg.junction_tree(edgeset)
    jt_sym_probs, jt_sym_acc = agg.junction_tree(edgeset, symmetric=True)


    print(f"Majority vote: {mv_acc}")
    print(f"Pick the best: {pb_acc}")
    print(f"Naive bayes: {accuracy_score(test_gold, np.array(nb_probs).argmax(axis=1))}")
    print(f"FlyingSquid: {accuracy_score(test_gold, np.array(fs_probs).argmax(axis=1))}")
    print(f"Junction tree (with deps): {accuracy_score(test_gold, np.array(jt_probs).argmax(axis=1))}")
    print(f"Junction tree (with deps, symmetric): {accuracy_score(test_gold, np.array(jt_sym_probs).argmax(axis=1))}")


    print(f"NB log loss {log_loss(test_gold, nb_probs)}")
    print(f"FS log loss {log_loss(test_gold, fs_probs)}")
    print(f"JT log loss {log_loss(test_gold, jt_probs)}")



    print(agg.get_probs(np.array([1, 1, 0]), agg.sym_accs, edgeset=[(0, 1)], symmetric=True, abstains_symmetric=False))
    print(agg.get_probs(np.array([1, 1, 0]), agg.nb_accs, edgeset=[(0, 1)], symmetric=False, abstains_symmetric=False))
    print(pgm.get_probs(np.array([1, 1, 0])))

    jt_probs = np.array(jt_probs)
    jt_sym_probs = np.array(jt_sym_probs)

    fail = False

    pgm_probs = np.zeros(len(test_votes))
    for i, votes in enumerate(test_votes):
        pgm_probs[i] = pgm.get_probs(votes)

    avg_jt_err = np.linalg.norm(pgm_probs - jt_probs[:, 1]) / n_test
    avg_jt_sym_err = np.linalg.norm(pgm_probs - jt_sym_probs[:, 1]) / n_test
    if avg_jt_err > 0.05:
        fail = True
    if avg_jt_err < avg_jt_sym_err:
        print(avg_jt_err, avg_jt_sym_err)
        fail= True


    if fail:
        print("Test failed.")
    else:
        print("Test passed.")

def test6():
    m = 3

    np.random.seed(5)
    thetas = np.random.rand(30)
    
    # model some edges - see if we can recover it
    potentials = [[3], [0, 3], [1, 3], [2, 3], [0, 1], [0, 2]]

    pgm = Ising(m, potentials, thetas)

    # make big datasets
    n_train = 1000
    train_votes, train_gold = pgm.make_data(n_train)
    n_test = 1000
    test_votes, test_gold = pgm.make_data(n_test)
    train_votes = to01(train_votes)
    train_gold = to01(train_gold)
    test_votes = to01(test_votes)
    test_gold = to01(test_gold)

    agg = Aggregator(train_votes, train_gold, test_votes, test_gold)


    # overall accuracies Pr(lf_p = y) on train
    acc_theta = np.zeros(m)
    for i in range(m):
        acc_theta[i] = len(np.where((train_votes[:, i] == train_gold) == True)[0])/n_train

    acc_theta = 2*acc_theta - 1
    all_thetas = structure_learning(m, train_votes, train_gold, acc_theta)
    print(all_thetas)

    #idx = np.argsort(all_thetas, axis=None)[-1]
    #i = int(np.round(idx / m))
    #j = idx % m
    #print(f"Recovered edge: ({i}, {j})") # should be (0, 1)

    ce = np.ones(m*m) * np.inf
    ce_cond = np.ones(m*m) * np.inf
    ce_nolabel = np.ones(m*m) * np.inf

    true_ce = np.ones(m*m) * np.inf
    true_ce_cond = np.ones(m*m) * np.inf
    true_ce_nolabel = np.ones(m*m) * np.inf


    neighborhood_size = len(all_thetas.flatten())

    all_edgesets = []

    for n in range(neighborhood_size):
        print(f"edgeset size is {n}")
        # try edgeset of size n
        if n != 0:
            idxs = np.argsort(np.abs(all_thetas), axis=None)[-n:]
            edgeset = []
            for idx in idxs:
                i = int(np.floor(idx / m))
                j = idx % m
                # print(all_thetas[i, j])
                # print(f"Recovered edge: ({i}, {j})") # should be (0, 1)
                edgeset.append((i, j))
        else:
            edgeset = []

        print(edgeset)
        all_edgesets.append(edgeset)
        try:
            ce[n] = agg.cross_entropy(train_votes, train_gold, edgeset)
            ce_cond[n] = agg.cross_entropy_conditional(train_votes, train_gold, edgeset)
            ce_nolabel[n] = agg.cross_entropy_no_label(test_votes, edgeset)

            true_ce[n] = pgm.cross_entropy(edgeset)
            true_ce_cond[n] = pgm.cross_entropy_conditional(edgeset)
            true_ce_nolabel[n] = pgm.cross_entropy_no_label(edgeset)

        except nx.NetworkXError:
            # skip if proposed graph is not triangulated
            pass

    print(ce)
    print(ce_cond)
    print(ce_nolabel)
    best_ce = ce.argmin()
    best_ce_cond = ce_cond.argmin()

    best_ce_nolabel = ce_nolabel.argmin()

    print(f"Recovered edgeset using MLE: {all_edgesets[best_ce]}")
    print(f"Recovered edgeset using MLE (conditional): {all_edgesets[best_ce_cond]}")

    print(f"Recovered edgeset using MLE (no labels): {all_edgesets[best_ce_nolabel]}")

    print(true_ce)
    print(true_ce_cond)

    print(true_ce_nolabel)


def main():
    #test0()
    #test1()
    #test2()
    #test3()
    test4()
    # test5()
    #test6()


if __name__ == "__main__":
    main()
