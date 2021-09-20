import math
import numpy as np
from sklearn import tree
from dataset import car_valuation
from sklearn.model_selection import learning_curve

import numpy as np
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

import warnings

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.exceptions import ConvergenceWarning

X = car_valuation.training_data
Y = car_valuation.training_label

# k_nn = KNeighborsClassifier()
# parameters = {'n_neighbors': list(range(1, 100)),
#               }
# gs = GridSearchCV(k_nn, parameters)
# gs.fit(X, Y)
# print(gs.best_params_)

clf = MLPClassifier()
hidden_layer_sizes = [(5, ) * i for i in range(1, 5)]
length_of_hidden_layers = [len(hl) for hl in hidden_layer_sizes]

train_scores, valid_scores = validation_curve(clf, X, Y, "hidden_layer_sizes", hidden_layer_sizes, cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(valid_scores, axis=1)
plt.plot(length_of_hidden_layers, train_scores_mean, label="train_scores")
plt.plot(length_of_hidden_layers, test_scores_mean, label="valid_scores")
plt.legend()
plt.title(f"Validation curve for NeuralNetwork (Num of Hidden Layers vs Accuracy)")
plt.savefig('Validation Curve.png')

plt.clf()

max_iter = 400
clf = MLPClassifier(hidden_layer_sizes=(5, ) * 2, max_iter=max_iter)
clf.fit(X, Y)


plt.plot(clf.loss_curve_)
plt.title(f"Learning curve for NeuralNetwork (Iteration vs Loss)")
plt.savefig('Learning Curve.png')

# for label, param in zip(labels, params):
#     print("training: %s" % label)
#     mlp = MLPClassifier(random_state=0,
#                         max_iter=max_iter, **param)
#
#     # some parameter combinations will not converge as can be seen on the
#     # plots so they are ignored here
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", category=ConvergenceWarning,
#                                 module="sklearn")
#         mlp.fit(X, y)
#
#     mlps.append(mlp)
#     print("Training set score: %f" % mlp.score(X, y))
#     print("Training set loss: %f" % mlp.loss_)
# for mlp, label, args in zip(mlps, labels, plot_args):
#     ax.plot(mlp.loss_curve_, label=label, **args)


predicted_label = clf.predict(car_valuation.test_data)
print(f"Accuracy of test data {accuracy_score(predicted_label, car_valuation.test_label)}")
