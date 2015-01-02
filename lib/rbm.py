__author__ = 'TJ Vandal'
'''
Much of this comes from Hinton's work and matlab code.
Sklearn RBM is used but some functions have been updated significantly to include sparsity and more options.

Notes:
Lots of imports
Comments coming
sklearn.neural_network.rbm.BernoulliRBM has some methods that I want access to
'''

import sys
import time

import numpy
from matplotlib import pyplot
from sklearn.neural_network.rbm import BernoulliRBM
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import check_random_state, check_arrays
from sklearn.utils.fixes import expit
import random
from matplotlib import cm


## This will train a single layer, with or without labels
class RBM(BernoulliRBM):
    def __init__(self, n_components=256, learning_rate=0.1, batch_size=10,
                 n_iter=10, verbose=0, random_state=None, learning_rate_bias=0.1,
                 regularization_mu=None, weight_cost=0.0002, phi=0.5, plot_weights=False,
                 plot_histograms=False, plot_reconstructed=False):
        BernoulliRBM.__init__(self, n_components, learning_rate, batch_size,
                              n_iter, verbose, random_state=random_state)
        if learning_rate_bias is not None:
            self.learning_rate_bias = learning_rate_bias
        else:
            self.learning_rate_bias = learning_rate
        self.weight_cost = weight_cost
        self.regularization_mu = regularization_mu
        self.cd_iter = int(n_iter / 5) + 1
        self.phi = phi

        self.plotweights = plot_weights
        self.plothist = plot_histograms
        self.plot_reconstructed = plot_reconstructed
        if self.regularization_mu is not None and (self.regularization_mu <= 0 or self.regularization_mu >= 1):
            err = "Regularization (%s) must between 0 and 1" % str(self.regularization_mu)
            raise ValueError(err)

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
        rng = check_random_state(self.random_state)

        ## initalize components and bias units
        num_cases, num_dim = X.shape
        numpy.random.seed(1000)

        self.components_ = numpy.asarray(numpy.random.normal(0, 0.1, size=(num_dim, self.n_components)))

        self.intercept_hidden_ = numpy.zeros((self.n_components,), dtype=float)
        self.intercept_visible_ = numpy.zeros((num_dim,), dtype=float)

        self.vishidinc = numpy.zeros((num_dim, self.n_components), dtype=float)
        self.hidbiasinc = numpy.zeros(self.intercept_hidden_.shape, dtype=float)
        self.visbiasinc = numpy.zeros(self.intercept_visible_.shape, dtype=float)

        ## If no target is given, set labels all to zero
        if targets is None:
            targets = numpy.zeros((num_cases, 1))
            backprop = False
        else:
            backprop = True
        num_labels = targets.shape[1]
        targets = numpy.array(targets)

        ## set components and bias of targets all to zero so that when targets are not given these biases
        # aren't taken into account. this is kind of a hack but works for the time being.
        self.target_components_ = numpy.asarray(rng.normal(0, 0.1, (num_labels, self.n_components)))
        self.target_bias_ = numpy.zeros((num_labels,))

        batch_slices = generate_random_batches(targets, self.batch_size)

        verbose = self.verbose
        verbose_freq = int(self.n_iter/10)
        begin = time.time()
        self.error_terms = numpy.zeros(self.n_iter, dtype=float)
        for iteration in xrange(1, self.n_iter + 1):
            errsum = 0
            for batch_slice in batch_slices:
                err = self._fit(X[batch_slice], rng, targets[batch_slice], epoch_num=iteration, backprop=backprop)
                errsum += err

            self.error_terms[iteration-1] = errsum
            if verbose and verbose_freq > 0 and (iteration % verbose_freq == 0 or iteration == 1):
                if self.plothist:
                    #h_pos_ = self._mean_hiddens(X)
                    for j, arr in enumerate([self.components_, self.intercept_visible_, self.intercept_hidden_]):
                        pyplot.subplot2grid((5, 1), (j, 0))
                        plot_histogram(arr, bin_count=50)

                    pyplot.draw()
                    pyplot.subplot2grid((5, 1), (3, 0), rowspan=2)
                    self.plot_hidden_units(X)

                if self.plotweights:
                    self.plot_weights(16)

                if self.plot_reconstructed:
                    self.plot_reconstructed(X[:10], targets=targets[:10])

                end = time.time()
                print("[%s] Iteration %d, SSE = %.2f, time = %.2fs"
                      % (type(self).__name__, iteration, errsum, end - begin))
                begin = end

        return self

    def _fit(self, v_pos, rng, targets, epoch_num=1, backprop=False):
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
        momentum = 0.7

        ######## VISABLE POSITIVE TO HIDDEN POSITIVE PHASE  ##########
        h_pos = self._mean_hiddens(v_pos, targets)
        h_pos_states = h_pos > numpy.random.random(h_pos.shape)
        if self.regularization_mu is not None:
            ## force sparsity by pressuring hidden units to turn on ##
            sparse_h_pos = self.regularization(h_pos, 1.-self.regularization_mu, axis=0)
            h_pos = self.phi * sparse_h_pos + (1 - self.phi) * h_pos
            h_pos_states = h_pos

        ######## HIDDEN POSITIVE TO VISABLE NEGATIVE PHASE  ##########
        batch_count = len(v_pos)
        self.target_bias_matrix = numpy.tile(self.target_bias_, (batch_count, 1))

        self.visible_bias_matrix = numpy.tile(self.intercept_visible_, (batch_count, 1))
        self.hidden_bias_matrix = numpy.tile(self.intercept_hidden_, (batch_count, 1))
        neg_lab_states = numpy.zeros(self.target_bias_matrix.shape, dtype=float)

        if backprop:
            temp_h_pos = h_pos
            pos_hid_states = h_pos_states
            for j in range(self.cd_iter):
                ## positive hidden label states from previous positive hidden probabilities
                neg_lab_prob = numpy.exp(numpy.dot(pos_hid_states, self.target_components_.T) + self.target_bias_matrix)
                neg_lab_prob = neg_lab_prob / neg_lab_prob.sum(axis=1).reshape(batch_count, 1)
                cum_probs = numpy.cumsum(neg_lab_prob, axis=1)
                sampling = cum_probs > rng.uniform(0, 1., (batch_count, 1))

                neg_lab_states = numpy.zeros(self.target_bias_matrix.shape, dtype=float)
                for j, s in enumerate(sampling):

                    # if unlabeled then don't predict state for training
                    if sum(targets[j]) == 0:
                        continue

                    try:
                        index = min(numpy.where(s)[0])
                        neg_lab_states[j, index] = 1

                    except ValueError:
                        for j, arr in enumerate([self.target_components_, self.intercept_hidden_, self.intercept_visible_]):
                            pyplot.subplot2grid((5, 1), (j, 0))
                            plot_histogram(arr, bin_count=50)
                        pyplot.draw()
                        time.sleep(20)
                        print "Failed..."
                        sys.exit(1)

                v_neg = expit(numpy.dot(pos_hid_states, self.components_.T) + self.intercept_visible_)
                v_neg = v_neg > numpy.random.uniform(0., 1., v_neg.shape)   ## given _sample_visibles this line is not needed
                temp_h_pos = self._mean_hiddens(v_neg, neg_lab_states)
                if self.regularization_mu is not None:
                    ## force sparsity by pressuring hidden units to turn on ##
                    sparse_h_pos = self.regularization(temp_h_pos, 1-self.regularization_mu, axis=0)
                    temp_h_pos = self.phi * sparse_h_pos + (1 - self.phi) * temp_h_pos
                    pos_hid_states = temp_h_pos > numpy.random.random(h_pos.shape)
                else:
                    pos_hid_states = temp_h_pos > numpy.random.random(h_pos.shape)

            h_neg = temp_h_pos

        else:
            ## visable negative must be a function of hidden positive states
            v_neg = expit(numpy.dot(h_pos_states, self.components_.T) + self.intercept_visible_)

            ######## VISABLE NEGATIVE TO HIDDEN NEGATIVE PHASE #########
            h_neg = expit(numpy.dot(v_neg, self.components_) + self.intercept_hidden_)

        err = numpy.sum(numpy.square((v_neg - v_pos)))

        ## compute learning rates by dividing by batch size
        lr = float(self.learning_rate) / v_pos.shape[0]
        lr_bias = float(self.learning_rate_bias) / v_pos.shape[0]

        ######## Update Components and Bias Units ########
        update_comp = numpy.dot(v_pos.T, h_pos)
        update_comp -= numpy.dot(v_neg.T, h_neg)
        update_comp -= self.weight_cost * v_pos.shape[0] * self.components_  # weight decay
        self.vishidinc = lr * update_comp + self.vishidinc * momentum
        self.components_ += self.vishidinc

        update_comp_lab = numpy.dot(targets.T, h_pos)
        update_comp_lab -= numpy.dot(neg_lab_states.T, h_neg)
        update_comp_lab -= self.weight_cost * v_neg.shape[0] * self.target_components_

        self.target_components_ += lr * update_comp_lab

        self.hidbiasinc = momentum * self.hidbiasinc + lr_bias * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        self.intercept_hidden_ += self.hidbiasinc

        self.visbiasinc = momentum * self.visbiasinc + lr_bias * (v_pos.sum(axis=0) - v_neg.sum(axis=0))
        self.intercept_visible_ += self.visbiasinc

        self.target_bias_ += lr_bias * (targets.sum(axis=0) - neg_lab_states.sum(axis=0))

        #print targets[:10]
        #print neg_lab_states[:10], "\n\n"

        return err

    def regularization(self, P, mu=0.1, sigma=0.0001, axis=1):
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
        ranks = 1 / (1 + numpy.exp(-(ranks - mu)/sigma))
        #ranks = ranks ** (1 / mu - 1)
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
        p = safe_sparse_dot(v, self.components_)
        p += self.intercept_hidden_

        if targets is not None:
            p += safe_sparse_dot(targets, self.target_components_)

        return 1. / (1. + numpy.exp(-p))

    def _log_binomial_sparsity(self, P, mu=0.2):
        penalty = - mu * numpy.log(P) - (1 - mu) * numpy.log(1 - P)
        P += penalty
        return P

    def plot_weights(self, num):
        num = min([len(self.components_), num])
        size = numpy.sqrt(self.components_.shape[0])
        if int(size) != size:
            return

        h = numpy.ceil(numpy.sqrt(num))
        w = numpy.ceil(num / h)
        if w != int(w):
            return

        indices = range(0, self.components_.shape[1])
        random.shuffle(indices)

        normalized_components = (self.components_ - self.components_.min()) / (self.components_.max() - self.components_.min())
        for i in range(num):
            pyplot.subplot(w, h, i + 1)
            X = normalized_components[:, random.randint(0, normalized_components.shape[1]-1)].reshape(size, size).T
            pyplot.imshow(X, cmap=cm.Greys, hold=False, vmin=0, vmax=1)
        pyplot.draw()

    def plot_hidden_units(self, vis, targets=None):
        hid = self._mean_hiddens(vis[:int(self.n_components/4)], targets)
        pyplot.imshow(hid, cmap=cm.Greys_r)
        pyplot.draw()

    def plot_reconstructed(self, v_pos, targets=None):
        h_pos = self._mean_hiddens(v_pos, targets)
        h_pos_states = h_pos > numpy.random.random(h_pos.shape)
        p = numpy.dot(h_pos_states, self.components_.T) + self.intercept_visible_
        v_neg = 1. / (1. + numpy.exp(-p))
        size = numpy.sqrt(v_neg.shape[1])
        i = numpy.random.randint(0, v_pos.shape[0]-1)

        if size == int(size):
            reconstructed = v_neg[i].reshape((size, size))
            img = v_pos[i].reshape((size, size))

            pyplot.subplot(1, 2, 1)
            pyplot.imshow(img.T, cmap=cm.Greys_r)
            pyplot.draw()
            pyplot.subplot(1, 2, 2)
            pyplot.imshow(reconstructed.T, cmap=cm.Greys_r)
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

    batch_slices = []
    for j in range(int(total_count / batch_size)):
        slice = []
        for col in increments.keys():
            slice += target_rows[col][j*batch_size:j*batch_size+increments[col]]
        if len(slice) != 0:
            batch_slices.append(slice)

    return batch_slices


def generate_random_batches(targets, batch_size):
    num_case = targets.shape[0]
    indices = range(num_case)
    random.shuffle(indices)

    slices = []
    for j in range(int(num_case/batch_size)):
        slices.append(indices[j*batch_size:(j+1)*batch_size])
    return slices


def plot_histogram(data, bin_count=50):
    hist, bins = numpy.histogram(data, bins=bin_count)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    pyplot.bar(center, 1.*hist/sum(hist), align='center', width=width, hold=False)
