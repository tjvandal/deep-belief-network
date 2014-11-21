__author__ = 'thomas.vandal'
'''
Notes:
Lots of imports
Comments coming
sklearn.neural_network.rbm.BernoulliRBM has some methods that I want access to
'''

import os, sys
import time

import numpy
from matplotlib import pyplot
from sklearn.neural_network.rbm import BernoulliRBM
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import check_random_state, check_arrays
from sklearn.utils.fixes import expit
import random

## This will train a single layer, with or without labels
class RBM(BernoulliRBM):
    def __init__(self, n_components=256, learning_rate=0.1, batch_size=10,
                 n_iter=10, verbose=0, random_state=None, learning_rate_bias=0.1,
                 regularization_mu=0.20, weight_cost=0.00005):
        BernoulliRBM.__init__(self, n_components, learning_rate, batch_size,
                              n_iter, verbose, random_state=None)
        if learning_rate_bias is not None:
            self.learning_rate_bias = learning_rate_bias
        else:
            self.learning_rate_bias = learning_rate
        self.weight_cost = weight_cost
        self.regularization_mu = regularization_mu
        self.cd_iter = int(n_iter / 20) + 1
        if self.regularization_mu is not None and (self.regularization_mu <= 0 or self.regularization_mu >= 1):
            raise ValueError("Regularization must between 0 and 1")

    def fit(self, X, targets=None):
        """Fit the model to the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Training data.
        target:  {array-like} shape (n_samples, n_unique_labels)

        Returns
        -------
        self : RBM
            The fitted model.
        """
        X[numpy.isnan(X)] = 0
        X, = check_arrays(X, sparse_format='csr', dtype=numpy.float)
        n_samples = X.shape[0]
        rng = check_random_state(self.random_state)

        ## initalize components and bias units
        num_cases, num_dim = X.shape
        RBM.components_ = numpy.asarray(rng.normal(0, 0.01, (self.n_components, num_dim)), order='fortran')
        RBM.intercept_hidden_ = numpy.random.normal(0, 0.01, (self.n_components,))
        RBM.intercept_visible_ = numpy.random.normal(0, 0.01, (X.shape[1],))


        ## If no target is given, set labels all to zero
        if targets is None:
            targets = numpy.zeros((num_cases, 1))

        num_labels = targets.shape[1]
        target = numpy.array(targets)

        ## set components and bias of targets all to zero so that when targets are not given these biases
        # aren't taken into account. this is kind of a hack but works for the time being.
        RBM.target_components_ = numpy.random.normal(0, 0.1, (self.n_components, num_labels))
        RBM.target_bias_ = numpy.zeros((num_labels,))

        ## split data into batches
        n_batches = int(numpy.ceil(float(n_samples) / self.batch_size))
        #batch_slices = list(gen_even_slices(n_batches * self.batch_size,
        #                                    n_batches, n_samples))

        batch_slices = generate_batch_slices(targets, self.batch_size)

        verbose = self.verbose
        begin = time.time()
        for iteration in xrange(1, self.n_iter + 1):
            for batch_slice in batch_slices:
                self._fit(X[batch_slice], rng, targets[batch_slice])

            if verbose and iteration % 2 == 0:
                for j, arr in enumerate([self.components_, self.intercept_hidden_, self.intercept_visible_]):
                    hist, bins = numpy.histogram(arr, bins=50)
                    pyplot.subplot(3, 1, j+1)
                    width = 0.7 * (bins[1] - bins[0])
                    center = (bins[:-1] + bins[1:]) / 2
                    pyplot.bar(center, hist, align='center', width=width, hold=False)
                    pyplot.draw()


                #self.plot_weights(16)

                end = time.time()
                print("[%s] Iteration %d, pseudo-likelihood = %.2f, time = %.2fs"
                      % (type(self).__name__, iteration, self.score_samples(X).mean(), end - begin))
                # print "Sum of Target Bias", self.target_bias_.sum()
                # print "Sum of Hidden Bias", self.intercept_hidden_.sum()
                # print "Sum of Visible Bias", self.intercept_visible_.sum()
                begin = end

        return self

    def _fit(self, v_pos, rng, targets):
        """Inner fit for one mini-batch.

        Adjust the parameters to maximize the likelihood of v using
        Stochastic Maximum Likelihood (SML).

        Parameters
        ----------
        v_pos : array-like, shape (n_samples, n_features)
            The data to use for training.

        rng : RandomState
            Random number generator to use for sampling.

        target: array-like, shape (n_samples, unique_labels)
            Labeled data.
        """

        ######## VISABLE POSITIVE TO HIDDEN POSITIVE PHASE  ##########
        h_pos = self._mean_hiddens(v_pos, targets)
        vvvv = v_pos
        ######## HIDDEN POSITIVE TO VISABLE NEGATIVE PHASE  ##########
        batch_count = len(v_pos)
        self.target_bias_matrix = numpy.tile(self.target_bias_, (batch_count, 1))
        self.visible_bias_matrix = numpy.tile(self.intercept_visible_, (batch_count, 1))
        self.hidden_bias_matrix = numpy.tile(self.intercept_hidden_, (batch_count, 1))
        neg_lab_states = self.target_bias_matrix * 0.

        if targets.sum() != 0:
            temp_h_neg = h_pos
            for j in range(self.cd_iter):
                neg_lab_states = self.target_bias_matrix * 0.
                ## positive hidden label states from previous positive hidden probabilities
                h_neg_states = h_pos > numpy.random.uniform(0., 1., size=h_pos.shape)

                neg_lab_prob = numpy.exp(numpy.dot(h_neg_states, self.target_components_) + self.target_bias_matrix)
                neg_lab_prob = neg_lab_prob / neg_lab_prob.sum(axis=1).reshape(batch_count, 1)

                cum_probs = numpy.cumsum(neg_lab_prob, axis=1)
                sampling = cum_probs > numpy.random.uniform(0, 1., (batch_count, 1))
                for j, s in enumerate(sampling):
                    try:
                        index = min(numpy.where(s)[0])
                        neg_lab_states[j, index] = 1
                    except ValueError:
                        print "v_pos", v_pos[j]
                        print "targets", targets[j]
                        print "h_neg_states", h_neg_states[j]
                        print "h_pos", h_pos[j]
                        sys.exit(1)

                v_neg = self._sample_visibles(h_neg_states, rng)
                v_neg_states = v_neg > numpy.random.uniform(0, 1, v_neg.shape)
                temp_h_neg = self._mean_hiddens(v_neg_states)
            h_neg = temp_h_neg
        else:
            if self.regularization_mu is not None:
                ## force sparsity by pressuring hidden units to turn on ##
                P_pos = self.selected_regularization(h_pos, mu=self.regularization_mu, axis=0)
                h_pos = self.phi * P_pos + (1 - self.phi) * h_pos

            ## compute hidden positive states
            h_pos_states = h_pos > numpy.random.uniform(0., 1., size=h_pos.shape)

            ## visable negative must be a function of hidden positive states
            v_neg = self._sample_visibles(h_pos_states, rng)

            ######## VISABLE NEGATIVE TO HIDDEN NEGATIVE PHASE #########
            h_neg = self._mean_hiddens(v_neg)

        ## compute learning rates by dividing by batch size
        lr = float(self.learning_rate) / v_pos.shape[0]
        lr_bias = float(self.learning_rate_bias) / v_pos.shape[0]

        ######## Update Components and Bias Units ########
        update_comp = safe_sparse_dot(v_pos.T, h_pos, dense_output=True)
        update_comp -= numpy.dot(v_neg.T, h_neg)
        update_comp -= self.weight_cost * v_pos.shape[0] * self.components_.T   # weight decay
        self.components_ += lr * update_comp.T

        update_comp_lab = safe_sparse_dot(targets.T, h_pos, dense_output=True)
        update_comp_lab -= numpy.dot(neg_lab_states.T, h_neg)
        update_comp_lab -= self.weight_cost * v_neg.shape[0] * self.target_components_.T
        self.target_components_ += lr * update_comp_lab.T

        self.intercept_hidden_ += lr_bias * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        self.intercept_visible_ += lr_bias * (v_pos.sum(axis=0) - v_neg.sum(axis=0))
        self.target_bias_ += lr_bias * (targets.sum(axis=0) - neg_lab_states.sum(axis=0))


    def selected_regularization(self, P, mu=0.1, axis=1):
        """
        http://www-poleia.lip6.fr/~cord/Publications_files/NIPS2010_Goh.pdf

        P is a numpy array exists [0, 1]
        """

        ranks = P.argsort(axis=axis).astype(float)      # + numpy.arange(P.shape[axis]).reshape(P.shape[axis,1])
        min_ranks = ranks.min(axis=axis).reshape(ranks.shape[axis - 1], 1)
        max_ranks = ranks.max(axis=axis).reshape(ranks.shape[axis - 1], 1)
        if axis == 0:
            min_ranks = min_ranks.T
            max_ranks = max_ranks.T

        ranks = (ranks - min_ranks) / (max_ranks - min_ranks)
        ranks = ranks ** (1 / mu - 1)
        return ranks

    def _mean_hiddens(self, v, targets=None):
        """Computes the probabilities P(h=1|v).

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        h : array-like, shape (n_samples, n_components)
            Corresponding mean field values for the hidden layer.
        """
        p = safe_sparse_dot(v, self.components_.T)
        p += self.intercept_hidden_
        if targets is not None:
            p += safe_sparse_dot(targets, self.target_components_.T)
        return expit(p, out=p)

    def _log_binomial_sparsity(self, P, mu=0.2):
        penalty = - mu * numpy.log(P) - (1 - mu) * numpy.log(1 - P)
        print P.min(), P.max(), numpy.log(P).min(), numpy.log(P).max()
        P += penalty
        return penalty

    def plot_weights(self, num):
        from matplotlib import cm
        random.seed(3)
        num = min([len(self.components_), num])
        size = numpy.sqrt(len(self.components_[0]))
        h = numpy.ceil(numpy.sqrt(num))
        w = numpy.ceil(num / h)


        indices = [random.randint(0, len(self.components_)-1) for j in range(num)]
        imgs = numpy.array([self.components_[i].reshape(size, size) for i in indices])
        imgs = imgs.reshape(h, w, size, size).swapaxes(1, 2).reshape((h*size, w*size))

        # close any open plots
        pyplot.close()
        pyplot.imshow(imgs, cmap=cm.Greys_r, hold=False)
        pyplot.show()

        for i in range(num):
            pyplot.subplot(w, h, i + 1)
            X = self.components_[random.randint(0, len(self.components_)-1)].reshape(size, size)
            pyplot.imshow(X, cmap=cm.Greys_r, hold=False)
            pyplot.draw()

## assumption, each row only has a single true vaue
def generate_batch_slices(targets, batch_size):
    target_counts = targets.sum(axis=0)
    total_count = targets.shape[0]
    target_rows = {}
    increments = {}
    cumulative_increments = 0.
    for col in range(targets.shape[1]):
        target_rows[col] = numpy.where(targets[:, col] == 1)[0].tolist()
        increments[col] = int(batch_size * target_counts[col] * 1. / total_count)
        cumulative_increments += increments[col]

    target_rows[-1] = numpy.where(targets.sum(axis=1) == 0)[0].tolist()
    increments[-1] = int(batch_size - cumulative_increments)

    #increments = {-1: 0, 0: 50, 1: 50}

    batch_slices = []
    for j in range(int(total_count / batch_size)):
        slice = []
        for col in increments.keys():
            slice += target_rows[col][j*batch_size:j*batch_size+increments[col]]
        if len(slice) != 0:
            batch_slices.append(slice)

    return batch_slices