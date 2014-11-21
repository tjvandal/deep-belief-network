__author__ = 'thomas.vandal'

from rbm import RBM
import numpy

class DeepBeliefNet:
    def __init__(self, num_layers, components, batch_size, learning_rate, bias_learning_rate, epochs, sparsity_rate=0.9):
        self.sparsity_rate = sparsity_rate
        try:
            self.num_layers = int(num_layers)
        except TypeError:
            raise TypeError("Number of layers must be an a number.")

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

            model = RBM(n_components=self.components[j], batch_size=self.batch_size[j],
                        learning_rate=self.learning_rate[j], regularization_mu=self.sparsity_rate,
                        n_iter=self.epochs[j], verbose=True, learning_rate_bias=self.bias_learning_rate[j])

            if j + 1 == self.num_layers and labels is not None:
                model.fit(temp_X, labels)
            else:
                model.fit(temp_X)

            temp_X = model.transform(temp_X)
            print "Trained Layer %i\n" % (j + 1)

            DeepBeliefNet.layers.append(model)

    def results(self, test_data, test_labels, label_column):
        from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
        layer_data = test_data
        for layer in self.layers[:-1]:
            print "layerdata: ", layer_data.shape, "\tcomponents:", layer.components_.T.shape, "\tintercepts:", layer.intercept_hidden_.shape
            p = numpy.dot(layer_data, layer.components_.T) + layer.intercept_hidden_
            layer_data = 1 / (1 + numpy.exp(-p))

        inter = numpy.zeros(test_labels.shape)
        layer = self.layers[self.num_layers - 1]
        for j in range(test_labels.shape[1]):
            targets = numpy.zeros(test_labels.shape)
            targets[:, j] = 1

            vis_bias_sum = numpy.dot(layer_data, layer.intercept_visible_.T) + numpy.dot(targets, layer.target_bias_.T)
            prod = layer.intercept_hidden_.T + numpy.dot(layer_data, layer.components_.T) \
                  + numpy.dot(targets, layer.target_components_.T)

            inter[:, j] = numpy.sum(numpy.log(1 + numpy.exp(prod)), axis=1) + vis_bias_sum

        max_row = inter.argmax(axis=1)
        temp = test_labels.argmax(axis=1)
        prediction = numpy.zeros(test_labels.shape[0])

        ## maxrow is the argmax which gives us the column with had the largest value
        # if this is equal to the test column than the prediction is true
        print "max row", max_row
        #prediction[label_column == max_row] = 1
        #print "inter", inter
        print "histgram of prediction", numpy.histogram(max_row)
        #print test_labels
        print "percentage classified correctly", sum(test_labels[:, label_column] == max_row) * 1.0 / len(temp)


        fpr, tpr, thresholds = roc_curve(test_labels[:, label_column], prediction)
        auc = roc_auc_score(test_labels[:, label_column], prediction)
        precision, recall, pr_thres = precision_recall_curve(test_labels[:, label_column], prediction)


        return {"fpr": fpr, "tpr": tpr, "auc": auc, "precision": precision, "recall": recall}

    def save_network(self, writefile):
        import pickle

        pickle.dump(self, open(writefile, "w"))

