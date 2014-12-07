import sys
import os

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(repo_dir, "src/python-mnist"))
sys.path.append(os.path.join(repo_dir, "lib"))

import pandas
from mnist import MNIST
from dbn import DeepBeliefNet
import numpy
from matplotlib import pyplot
import simplejson
import requests
import gzip
import StringIO

pyplot.ion()
pyplot.show()

mndata_file = os.path.join(repo_dir, "examples/data/mnist_data.json")
mndata_dir = os.path.join(repo_dir, "examples/data/")
mnist_data_files = {'training_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                    'training_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                    'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                    'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'}

## if files are not in local directory then download and decompress them
for key in mnist_data_files:
    url = mnist_data_files[key]
    filename = os.path.basename(url)
    if filename not in os.listdir(mndata_dir):
        r = requests.get(mnist_data_files[key], stream=True)
        compressed_file=StringIO.StringIO()
        compressed_file.write(r.content)
        compressed_file.seek(0)
        decompressed = gzip.GzipFile(fileobj=compressed_file, mode='rb')
        with open(os.path.join(mndata_dir, filename.split(".")[0]),'wb') as handle:
            handle.write(decompressed.read())

mndata = MNIST(mndata_dir)

if os.path.exists(mndata_file):
    js = simplejson.load(open(mndata_file, 'r'))
    mndata.train_images = js['train_data']
    mndata.train_labels = js['train_labels']
    mndata.test_images = js['test_data']
    mndata.test_labels = js['test_labels']

else:
    mndata.load_training()
    mndata.load_testing()
    js = {"train_data": mndata.train_images, "train_labels": mndata.train_labels.tolist(),
          "test_data": mndata.test_images, "test_labels": mndata.test_labels.tolist()}

    simplejson.dump(js, open(mndata_file, 'w'))

size = numpy.sqrt(len(mndata.train_images[0]))

mndata.train_images = numpy.array(mndata.train_images)
mndata.test_images = numpy.array(mndata.test_images)

mndata.train_images = mndata.train_images / 255.
mndata.test_images = mndata.test_images / 255.


mndata.train_labels = numpy.array(mndata.train_labels)
temp_labels = numpy.zeros((mndata.train_images.shape[0], 10))
for j in range(temp_labels.shape[0]):
    temp_labels[j, mndata.train_labels[j]] = 1

mndata.train_labels = temp_labels

temp_labels = numpy.zeros((mndata.test_images.shape[0], 10))
for j in range(temp_labels.shape[0]):
    temp_labels[j, mndata.test_labels[j]] = 1

mndata.test_labels = temp_labels

img_num = 2000
test_num = 10000
layers = 3
components = [200, 400, 800]
batch_size = 25
learning_rate = 0.1
bias_learning_rate = 0.1
epochs = 100
sparsity = None

train_img = mndata.train_images[:img_num] > 0.5
train_lab = mndata.train_labels[:img_num]

test_img = mndata.test_images[:test_num] > 0.5
test_lab = mndata.test_labels[:test_num]

DBN = DeepBeliefNet(layers, components, batch_size, learning_rate, bias_learning_rate, epochs, sparsity_rate=sparsity)
DBN.plot_weights = False
DBN.plot_histograms = True
DBN.fit_network(train_img, train_lab)
## the label column will get roc and pr curves relatives to that column (one versus all). Here we look at the number 1
results = DBN.results(test_img, test_lab, label_column=1, write_file=os.path.join(repo_dir, "examples/results/results.json"))

pyplot.close()
df = pandas.DataFrame({"true positive rates": results['tpr'], "false positive rates": results['fpr']})
df.plot(x="false positive rates", y="true positive rates", figsize=(10, 6), linewidth=2)
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.title("AUC = %0.3f" % results['auc'])
pyplot.savefig(os.path.join(repo_dir, "examples/results/roc.svg"))