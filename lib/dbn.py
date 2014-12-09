__author__ = 'TJ Vandal'
'''
DeepBeliefNet chains Restricted Boltzmann Machines together and trains a Deep Belief Network.
The last layer is the only layer which will be trained with labels.

Works is based on Geoffrey Hinton's matlab code.
'''

from rbm import RBM
import numpy
import json

class DeepBeliefNet:
    #@TODO I don't like how this initalization is done!
    def __init__(self, num_layers, components, batch_size=100, learning_rate=0.1, bias_learning_rate=0.1, epochs=20,
                 sparsity_rate=None, phi_sparsity=0.99):
        try:
            self.num_layers = int(num_layers)
        except TypeError:
            raise TypeError("Number of layers must be an a number.")

        if len(sparsity_rate) == self.num_layers:
            self.sparsity_rate = sparsity_rate
        else:
            try:
                self.sparsity_rate = [sparsity_rate] * self.num_layers
            except TypeError:
                raise Exception("Number of Components (%i) must be equal to the number of layers, %i" %
                            (len(components), layers))

        try:
            self.components = [int(components)] * self.num_layers
        except TypeError:
            if len(components) != self.num_layers:
                raise Exception(
                    "Number of Components (%i) must be equal to the number of layers, %i" % (len(components), layers))
            self.components = components

        try:
            self.batch_size = [int(batch_size)] * self.num_layers
        except TypeError:
            if len(batch_size) != self.num_layers:
                raise Exception(
                    "Number of batches (%i) must be equal to the number of layers, %i" % (len(batch_size), layers))
            self.batch_size = batch_size

        try:
            self.learning_rate = [float(learning_rate)] * self.num_layers
        except TypeError:
            if len(learning_rate) != self.num_layers:
                raise Exception("Number of learning rates (%i) must be equal to the number of layers, %i" % (
                    len(learning_rate), layers))
            self.learning_rate = learning_rate

        try:
            self.bias_learning_rate = [float(bias_learning_rate)] * self.num_layers
        except TypeError:
            if len(bias_learning_rate) != self.num_layers:
                raise Exception("Number of Bias Learning Rates (%i) must be equal to the number of layers, %i" % (
                    len(bias_learning_rate), layers))
            self.bias_learning_rate = bias_learning_rate

        try:
            self.epochs = [int(epochs)] * self.num_layers
        except TypeError:
            if len(epochs) != self.num_layers:
                raise Exception(
                    "Number of Epochs (%i) must be equal to the number of layers, %i" % (len(epochs), layers))
            self.epochs = epochs

        self.plot_weights = False
        self.plot_histograms = False
        self.phi = phi_sparsity



    ## if labels are given then we will use them to train the top layer
    def fit_network(self, X, labels=None):
        if labels is None:
            labels = numpy.zeros((X.shape[0], 2))
        DeepBeliefNet.layers = []
        temp_X = X
        for j in range(self.num_layers):

            print "\nTraining Layer %i" % (j + 1)
            print "components: %i" % self.components[j]
            print "batch_size: %i" % self.batch_size[j]
            print "learning_rate: %0.3f" % self.learning_rate[j]
            print "bias_learning_rate: %0.3f" % self.bias_learning_rate[j]
            print "epochs: %i" % self.epochs[j]
            print "Sparsity: %s" % str(self.sparsity_rate[j])
            print "Sparsity Phi: %s" % str(self.phi)
            if j != 0:
                self.plot_weights = False

            model = RBM(n_components=self.components[j], batch_size=self.batch_size[j],
                        learning_rate=self.learning_rate[j], regularization_mu=self.sparsity_rate[j],
                        n_iter=self.epochs[j], verbose=True, learning_rate_bias=self.bias_learning_rate[j],
                        plot_weights=self.plot_weights, plot_histograms=self.plot_histograms, phi=self.phi)

            if j + 1 == self.num_layers and labels is not None:
                model.fit(numpy.asarray(temp_X), numpy.asarray(labels))
            else:
                model.fit(numpy.asarray(temp_X))

            temp_X = model._mean_hiddens(temp_X)  # hidden layer given visable units
            print "Trained Layer %i\n" % (j + 1)

            DeepBeliefNet.layers.append(model)

    def results(self, test_data, test_labels, label_column, write_file=None):
        from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
        layer_data = test_data
        for layer in self.layers[:-1]:
            p = numpy.dot(layer_data, layer.components_) + layer.intercept_hidden_
            layer_data = 1 / (1 + numpy.exp(-p))

        inter = numpy.zeros(test_labels.shape)
        layer = self.layers[self.num_layers - 1]
        for j in range(test_labels.shape[1]):
            targets = numpy.zeros(test_labels.shape)
            targets[:, j] = 1

            vis_bias_sum = numpy.dot(layer_data, layer.intercept_visible_.T) + numpy.dot(targets, layer.target_bias_.T)
            prod = layer.intercept_hidden_.T + numpy.dot(layer_data, layer.components_) + numpy.dot(targets, layer.target_components_)
            inter[:, j] = numpy.sum(numpy.log(1 + numpy.exp(prod)), axis=1) + vis_bias_sum
      
        normalized_inter = inter / inter.sum(axis=1).reshape(len(inter), 1)
        max_row = normalized_inter.argmax(axis=1)
        lab = test_labels.argmax(axis=1)

        print "histgram of prediction", numpy.histogram(max_row, bins=len(numpy.unique(max_row)))
        print "percentage classified correctly", sum(lab == max_row) * 1.0 / len(lab)

        fpr, tpr, thresholds = roc_curve(test_labels[:, label_column], normalized_inter[:, label_column])
        auc = roc_auc_score(test_labels[:, label_column], normalized_inter[:, label_column])
        precision, recall, pr_thres = precision_recall_curve(test_labels[:, label_column], normalized_inter[:, label_column])

        DeepBeliefNet.res = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "precision": precision.tolist(),
                             "recall": recall.tolist(), 'auc': auc}

        if write_file:
            with open(write_file, 'w') as write:
                json.dump(DeepBeliefNet.res, write)

        return DeepBeliefNet.res

    def save_network(self, writefile):
        import pickle

        pickle.dump(self, open(writefile, "w"))

