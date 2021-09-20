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
from sklearn.neighbors import KNeighborsClassifier

X = car_valuation.training_data
Y = car_valuation.training_label

# k_nn = KNeighborsClassifier()
# parameters = {'n_neighbors': list(range(1, 100)),
#               }
# gs = GridSearchCV(k_nn, parameters)
# gs.fit(X, Y)
# print(gs.best_params_)

clf = KNeighborsClassifier()
estimators = list(range(1, 20))

train_scores, valid_scores = validation_curve(clf, X, Y, "n_neighbors", estimators, cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(valid_scores, axis=1)
plt.plot(estimators, train_scores_mean, label="train_scores")
plt.plot(estimators, test_scores_mean, label="valid_scores")
plt.legend()
plt.title(f"Validation curve for K-Neighbors (Number of Neighbors vs Accuracy)")
plt.savefig('Validation Curve.png')

plt.clf()

clf = KNeighborsClassifier(n_neighbors=5)
train_size = np.linspace(0.1, 1, 10)
train_sizes, train_scores, validation_scores = learning_curve(clf, X, Y, cv=5, train_sizes=train_size)
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(validation_scores, axis=1)


plt.plot(train_sizes, train_scores_mean, label="train_scores")
plt.plot(train_sizes, val_scores_mean, label="valid_scores")
plt.legend()
plt.title(f"Learning curve for K-Neighbors (Number of Neighbors vs Accuracy)")
plt.savefig('Learning Curve.png')

clf.fit(X, Y)
predicted_label = clf.predict(car_valuation.test_data)
print(f"Accuracy of test data {accuracy_score(predicted_label, car_valuation.test_label)}")
