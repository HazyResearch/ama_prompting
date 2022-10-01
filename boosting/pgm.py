import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy.stats



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

        self.support = np.array(list(map(list, itertools.product(vals, repeat=self.v))))

        self._make_pdf()
        self._make_cdf()

        self._get_means()
        self._get_balance()
        self._get_accs()

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

    # def _get_covariance(self):

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



def est_accs(m, vote, gold):
    # compute pr(lf | y) accuracies. Each prompt has 4 values (2x2)
    # we need to do this on the train/dev set
    classes = [0, 1]
    gold_idxs = [np.where(gold == -1)[0], np.where(gold == 1)[0]]

    accs = np.zeros((m, 2, 2)) # [i, j, k] = Pr(prompt_i = j| y = k)
    for p in range(m):
        for i in classes:
            for j in classes:
                accs[p, i, j] = len(np.where(vote[gold_idxs[i], p] == 2*j-1)[0]) / len(gold_idxs[i])

    return accs

def est_balance(gold, n):
    return len(np.where(gold == 1)[0]) / n

# Pr(lf votes, y)
def get_cond_probs(m, votes, y, accs, balance):
    pr_y = balance if y == 1 else 1 - balance
    prod = pr_y
    for i in range(m):
        prod *= accs[i, y, int(0.5*(votes[i] + 1))] # this assumes everything is independent
    return prod 

# Pr(y = 1 | lf votes)
def get_probs(m, votes, accs, balance):
    pos = get_cond_probs(m, votes, 1, accs, balance)
    neg = get_cond_probs(m, votes, 0, accs, balance)

    if pos == 0:
        return 0
    else:
        return pos / (pos + neg)


def pick_best_prompt(m, vote, gold, n):
    # overall accuracies Pr(lf_p = y) on test (we don't know these)
    overall_train_acc = np.zeros(m)
    for i in range(m):
        overall_train_acc[i] = len(np.where((vote[:, i] == gold) == True)[0])/n

    return overall_train_acc.argmax()


def main():

    # number of weak labels
    m = 5

    # total number of vertices
    v = m + 1

    # randomly parametrize exponential family to determine accuracies and correlations
    #theta = np.random.rand()
    #theta_cliques = (np.random.randint(0, 2, 5)*2 - 1)*theta
    #theta = np.random.rand()
    #theta_cliques = [1, 1, 1, 1, 1, 1, 1]
    thetas = np.random.rand(30)

    # all conditionally independent
    potentials = [[5], [0], [1], [4], [0, 5], [1, 5], [2, 5], [3, 5], [4, 5]]

    pgm = Ising(m, potentials, thetas)

    n_train = 10000
    vote_train, gold_train = pgm.make_data(n_train)

    n_test = 1000
    vote_test, gold_test = pgm.make_data(n_test)

    accs = est_accs(m, vote_train, gold_train)
    balance = est_balance(gold_train, n_train)

    nb_output = np.zeros(n_test) # naive bayes
    mv_output = np.zeros(n_test)

    nb_err = 0
    mv_err = 0

    for i in range(n_test):
        nb_output[i] = 2*np.round(get_probs(m, vote_test[i], accs, balance))-1
        if nb_output[i] != gold_test[i]:
            nb_err += 1

    
        # note: play around with MV tie breaking strategy 
        if len(np.where(vote_test[i] == 1)[0]) >= m / 2:
            mv_output[i] = 1
        elif len(np.where(vote_test[i] == 1)[0]) < m / 2:
            mv_output[i] = -1
        else:
            mv_output[i] = 2*np.random.randint(0, 2)-1

        if mv_output[i] != gold_test[i]:
            mv_err += 1

    nb_acc = 1 - (nb_err / n_test)
    mv_acc = 1 - (mv_err / n_test)
    #fs_acc = 1 - (fs_err / n_test)

    best_prompt = pick_best_prompt(m, vote_train, gold_train, n_train)

    best_prompt_acc = len(np.where((vote_test[:, best_prompt] == gold_test) == True)[0]) / n_test

    print(f"Naive bayes: {nb_acc}")
    print(f"Best prompt: {best_prompt_acc}")
    print(f"Majority vote: {mv_acc}")


if __name__ == "__main__":
   main()