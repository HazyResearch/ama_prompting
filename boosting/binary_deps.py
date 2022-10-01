import networkx as nx
import numpy as np
from itertools import chain, product, combinations
from scipy.sparse import issparse
import more_itertools
import torch


class DependentPGM:
    """
        This class describes a PGM learned from labeled data with specified edge structure.

        Args: 
            edges: list of edges that are dependent
            train_votes: n x m array of votes in {0, 1}
            train_gold: n array of true labels in {0, 1}
    """
    def __init__(
        self, edges, train_votes, train_gold, abstains = False, classes = [0, 1], abstain_value = -1) -> None:
        """
            Initialize the PGM by computing its junction tree factorization (c_tree and c_data)
            and by computing individual LF accuracy and class balance.
        """

        self.edges = edges
        self.train_votes = train_votes 
        self.train_gold = train_gold
        
        self.classes = classes
        self.k = len(classes)
        assert len(np.unique(self.train_gold)) == self.k 

        self.abstains = abstains
        assert len(np.unique(self.train_votes)) == int(abstains) + self.k 
        self.abstain_value = abstain_value

        self.n, self.m = self.train_votes.shape

        self.nodes = np.arange(self.m) 
        self.higher_order = len(edges) != 0

        # construct data structures containing dependency graph information (maximal cliques and separator sets)
        self._set_clique_tree()
        self._set_clique_data()

        # compute LF accuracies and class balance
        self._get_accs_and_cb()

    def _get_scaled(self):
        if self.classes == [0, 1]:
            self.train_votes_scaled = 2*self.train_votes - 1
            self.train_gold_scaled = 2*self.train_gold - 1
            if self.abstains:
                self.train_votes_scaled[self.train_votes == self.abstain_value] = 0
        else:
            self.train_votes_scaled = self.train_votes
            self.train_gold_scaled = self.train_gold
           




    def _set_clique_tree(self):
        G1 = nx.Graph()
        G1.add_nodes_from(self.nodes)
        G1.add_edges_from(self.edges)
        
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
        # Create a helper data structure which maps cliques (as tuples of member
        # sources) --> {start_index, end_index, maximal_cliques}, where
        # the last value is a set of indices in this data structure    
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
            for item in chain(self.c_tree.nodes(), self.c_tree.edges()):
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


    def _get_accs_and_cb(self):
        classes = [0, 1]
        self.gold_idxs = [np.where(self.train_gold == c)[0] for c in classes]
        
        self.accs = np.zeros((self.m, 2)) # [i, j, k] = Pr(prompt_i = j| y = k)
        for p in range(self.m):
            for i in classes:
                self.accs[p, i] = len(np.where(self.train_votes[self.gold_idxs[i], p] == 1)[0]) / len(self.gold_idxs[i])

        self.accs = np.clip(self.accs, 0.0001, 0.9999)       
        self.balance = len(self.gold_idxs[1]) / self.n

    def get_clique_probs(self, idxs, vals, y):
        """
            Computes marginal probability over voters indexed by idx, Pr(votes_idxs = vals | y).
        """
        truth_matrix = np.ones(len(self.gold_idxs[y])).astype(bool)
        for i, lf in enumerate(idxs):
            truth_matrix = np.logical_and(truth_matrix, self.train_votes[self.gold_idxs[y], lf] == vals[i])

        if len(np.where(truth_matrix == True)[0]) == 0:
            return 0.00001
        return len(np.where(truth_matrix == True)[0]) / len(self.gold_idxs[y])


    def get_cond_probs(self, votes, y):
        """
            Computes the probability Pr(votes | y).
        """
        pr_y = self.balance if y == 1 else 1 - self.balance
        prod = pr_y
        
        for i in self.c_tree.nodes():
            node = self.c_tree.nodes[i]
            members = list(node['members'])
            if len(members) == 1:
                v = members[0]
                print(f"multiplying by {votes[v] * self.accs[v, y]}")
                prod *= votes[v] * self.accs[v, y] + (1 - votes[v]) * (1 - self.accs[v, y])
            else:
                print(members)
                print(f"multiplying by {self.get_clique_probs(members, votes[members], y)}")

                prod *= self.get_clique_probs(members, votes[members], y)
                
        for i in self.c_tree.edges():
            edge = self.c_tree.edges[i]
            members = list(edge['members'])
            if len(members) == 1:
                v = members[0]
                deg = len(self.c_data[v]['max_cliques'])
                prod /= (votes[v] * self.accs[v, y] + (1 - votes[v]) * (1 - self.accs[v, y]))**(deg-1)

                print(members)
                print(f"Dividing by {votes[v] * self.accs[v, y] + (1 - votes[v]) * (1 - self.accs[v, y])} to the {deg - 1} power")

            else:
                deg = len(self.c_data[tuple(members)]['max_cliques'])
                prod /= (self.get_clique_probs(members, votes[members], y))**(deg-1)

                print(members)
                print(f"Dividing by {self.get_clique_probs(members, votes[members], y)} to the {deg - 1} power")

        print(prod)
        return prod 

    def get_probs(self, votes):
        """
            Computes the probability Pr(y = 1 | votes).
        """
        pos = self.get_cond_probs(votes, 1)
        neg = self.get_cond_probs(votes, 0)
        if pos == 0:
            return 0
        else:
            return pos / (pos + neg)

    def evaluate(self, test_votes, test_gold):
        """
            Using our learned PGM, output rounded estimates of Pr(y = 1 | votes) and computes its accuracy.

            Args:
                test_votes: vote array to perform inference on in {0, 1}
                test_gold: true labels to compare to in {0, 1}
        """
        n_test = len(test_votes)
        
        output_rounded = np.zeros(n_test)
        output_probs = np.zeros(n_test)
        err = 0
        for i in range(n_test):
            output_probs[i] = self.get_probs(test_votes[i])
            output_rounded[i] = np.round(output_probs[i])
            err += output_rounded[i] != test_gold[i]

        accuracy = 1 - err / n_test

        return output_probs, output_rounded, accuracy


def is_triangulated(nodes, edges):
    """
        If a graph is triangulated (e.g. if a junction tree factorization exists).
    """
    G1 = nx.Graph()
    G1.add_nodes_from(nodes)
    G1.add_edges_from(edges)
    return nx.is_chordal(G1)


def structure_learning(m, votes, gold, acc_theta, classes = [0, 1], l1_lambda=0.2):
    """
    Structure learning algorithm (Ising model selection) from Ravikumar (2010).

    Args:
    - votes: n_train x m array of training votes
    - gold: n_train array of gold labels on the training data
    - acc_theta: E[vote_i y] (where vote and y are scaled to [-1, 1]). This is a scaled version of accuracy that we will initialize some of the
    parameters in our PGM with in order to specify that we don't want to optimize over the edges between votes and y.
    We only are learning edges among votes!
    - classes: the list of classes the data can take on.
    - l1_lambda: l1 regularization strength
    """

    # scale the data
    classes = np.sort(np.unique(gold))
    vote_classes = np.sort(np.unique(votes))
    if 0 in classes and 1 in classes:
        votes_scaled = 2*votes - 1
        gold_scaled = 2*gold - 1
        if len(vote_classes) == len(classes) + 1:
            votes_scaled[votes == -1] = 0
    else:
        votes_scaled = votes 
        gold_scaled = gold

    acc_theta = torch.from_numpy(acc_theta).type(torch.FloatTensor) 
    all_thetas = np.zeros((m, m)) # learned thetas from alg

    # for each prompt, we fit a logistic regression model on it with prompt_i's output as the response variable and all otehr prompt outputs as the covariates.
    # big_theta is a vector of weights that denote dependence on each prompt (0 is independence).
    for v in range(m):
        print(f"Learning neighborhood of vertex {v}.")
        if len(classes) == 2:
            big_theta = learn_neighborhood(m, v, votes_scaled, gold_scaled, acc_theta, l1_lambda)
        else:
            big_theta = learn_neighborhood_multi(m, v, votes_scaled, gold_scaled, acc_theta, l1_lambda, classes)
        all_thetas[v] = big_theta

    return all_thetas


# v is the vertex whose neighborhood graph we are estimating 
def learn_neighborhood(m, vertex, votes, gold, accs, l1_lambda, epochs = 50000):
    """
    Learn the neighborhood graph for a vertex.

    Args:
    - m: number of prompts
    - vertex: the index of the prompt we are selecting as the response variable
    - votes: votes on training data
    - gold: gold label of training data
    - accs: training accuracies of each prompt we use to initialize the PGM parameters with 
    - l1_lambda: regularization strength
    - epochs: number of iterations
    """
    n = len(gold)
    vote_y = np.concatenate((votes, gold.reshape(n, 1)), axis=1)

    xr = vote_y[:, vertex]
    x_notr = np.delete(vote_y, vertex, axis=1)

    xr = torch.from_numpy(xr).type(torch.FloatTensor)
    x_notr = torch.from_numpy(x_notr).type(torch.FloatTensor)


    theta = torch.zeros(m) # last index is for accuracy between vertex and y 
    theta[m - 1] = accs[vertex] # initialize this to be the train accuracy. We do want this to be an optimizable variable still though.
    theta.requires_grad_()

    optimizer = torch.optim.SGD([theta], lr=0.0001)
    for t in range(epochs):
        optimizer.zero_grad()

        # logistic regression from Ravikumar et al
        fx = (torch.log(torch.exp(torch.matmul(x_notr, theta)) 
                        + torch.exp(-torch.matmul(x_notr, theta))).mean())
        loss = fx - torch.multiply(xr, x_notr.T).mean(dim=1).dot(theta) + l1_lambda * torch.linalg.vector_norm(theta[:m], ord=1)

        loss.backward()
        optimizer.step()

        #if t % 1000 == 0:
        #    print(f"Loss: {loss}")

    big_theta = np.concatenate([theta.detach().numpy()[:vertex], [0], theta.detach().numpy()[vertex:m - 1]])
    return big_theta

# v is the vertex whose neighborhood graph we are estimating 
def learn_neighborhood_multi(m, vertex, votes, gold, accs, l1_lambda, classes, epochs = 50000):
    # votes: in range {0, ... k}
    n = len(gold)
    vote_y = np.concatenate((votes, gold.reshape(n, 1)), axis=1)

    xr = vote_y[:, vertex]
    x_notr = np.delete(vote_y, vertex, axis=1)

    xr = torch.from_numpy(xr).type(torch.FloatTensor)
    x_notr = torch.from_numpy(x_notr).type(torch.FloatTensor)


    theta = torch.zeros(m) # last index is for accuracy between vertex and y 
    theta[m - 1] = accs[vertex] # initialize this 
    theta.requires_grad_()

    optimizer = torch.optim.SGD([theta], lr=0.0001)
    for t in range(epochs):
        optimizer.zero_grad()

        # logistic regression from Ravikumar et al
        mu = 0
        for i in range(x_notr.shape[1]):
            # mu = \sum_i theta_i * \sum_data sign{x_r = x_i}
            mu += (2*(xr == x_notr[:, i])-1).type(torch.FloatTensor).mean() * theta[i]

        fx = 0
        for k in classes:
            # \sum_y exp( \sum_i theta_i sign(x_i = y)) "normalization"
            fx += torch.exp(torch.matmul((2*(x_notr == k)-1).type(torch.FloatTensor), theta)).mean()
        
        loss = fx - mu + l1_lambda * torch.linalg.vector_norm(theta[:m], ord=1)

        loss.backward()
        optimizer.step()

        #if t % 1000 == 0:
        #    print(f"Loss: {loss}")

    big_theta = np.concatenate([theta.detach().numpy()[:vertex], [0], theta.detach().numpy()[vertex:m - 1]])
    return big_theta

def main():
    # load data
    vote_arr_train = np.load('./data/youtube-spam/train_votes.npy').T
    vote_arr_test = np.load('./data/youtube-spam/test_votes.npy').T
    gold_arr_train = np.load('./data/youtube-spam/train_gold.npy').T
    gold_arr_test = np.load('./data/youtube-spam/test_gold.npy').T

    # vote_arr_train = np.concatenate((vote_arr_train[:, 0: 2], vote_arr_train[:, 4:]), axis=1)
    # vote_arr_test = np.concatenate((vote_arr_test[:, 0: 2], vote_arr_test[:, 4:]), axis=1)

    n_train, num_prompts = vote_arr_train.shape


    # make validation set 
    np.random.seed(4)
    val_idxs = np.random.choice(np.arange(n_train), size= 28, replace=False)
    vote_arr_val = vote_arr_train[val_idxs, :]
    vote_arr_train = np.delete(vote_arr_train, val_idxs, axis=0)

    gold_arr_val = gold_arr_train[val_idxs]
    gold_arr_train = np.delete(gold_arr_train, val_idxs)

    nodes = np. arange(num_prompts)
    

    # specify edgeset
    # edges =[(0, 1)]
    #model = DependentPGM(edges, vote_arr_train, gold_arr_train)
    #probs, output, acc = model.evaluate(vote_arr_test, gold_arr_test)
    #print(acc)

    
    # Brute-force iteration through a bunch of edges 
    all_edges = list(combinations(nodes, 2))
    small_edgesets = list(more_itertools.powerset(all_edges))
    #small_edgesets = list(combinations(all_edges, 0)) + list(combinations(all_edges, 1)) + list(combinations(all_edges, 2)) + list(combinations(all_edges, 3))
    scores = np.zeros(len(small_edgesets))
    
    for i, edgeset in enumerate(small_edgesets):
        if len(edgeset) > 4:
            break
        if not is_triangulated(nodes, edgeset):
            continue
        model = DependentPGM(edgeset, vote_arr_train, gold_arr_train)
        
        probs, output, scores[i] = model.evaluate(vote_arr_val, gold_arr_val)
        if i % 100 == 0:
            print(f"Edgeset: {edgeset} \n score: {scores[i]}")

    print(f"Best edgeset score: {scores.max()}")
    print(f"Best edgeset: {small_edgesets[scores.argmax()]}")

    edges = small_edgesets[scores.argmax()]

    vote_arr_train = np.concatenate((vote_arr_train, vote_arr_val))
    gold_arr_train = np.concatenate((gold_arr_train, gold_arr_val))

    model = DependentPGM(edges, vote_arr_train, gold_arr_train)
    probs, output, acc = model.evaluate(vote_arr_test, gold_arr_test)
    print(f"Final model accuracy: {acc}")


if __name__ == "__main__":
   main()
