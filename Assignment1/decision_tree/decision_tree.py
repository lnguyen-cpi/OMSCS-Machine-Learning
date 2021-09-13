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

X = car_valuation.training_data
Y = car_valuation.training_label

# clf = clf.fit(X, Y)

# train_sizes, train_scores, valid_scores = learning_curve(
#     clf, X, Y, train_sizes=[0.3, 0.5, 0.8]
# )

# print(train_scores)
#
# print([sum(x) / len(x) for x in valid_scores])


# def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
#                         n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
#     """
#     Generate 3 plots: the test and training learning curve, the training
#     samples vs fit times curve, the fit times vs score curve.
#
#     Parameters
#     ----------
#     estimator : estimator instance
#         An estimator instance implementing `fit` and `predict` methods which
#         will be cloned for each validation.
#
#     title : str
#         Title for the chart.
#
#     X : array-like of shape (n_samples, n_features)
#         Training vector, where ``n_samples`` is the number of samples and
#         ``n_features`` is the number of features.
#
#     y : array-like of shape (n_samples) or (n_samples, n_features)
#         Target relative to ``X`` for classification or regression;
#         None for unsupervised learning.
#
#     axes : array-like of shape (3,), default=None
#         Axes to use for plotting the curves.
#
#     ylim : tuple of shape (2,), default=None
#         Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).
#
#     cv : int, cross-validation generator or an iterable, default=None
#         Determines the cross-validation splitting strategy.
#         Possible inputs for cv are:
#
#           - None, to use the default 5-fold cross-validation,
#           - integer, to specify the number of folds.
#           - :term:`CV splitter`,
#           - An iterable yielding (train, test) splits as arrays of indices.
#
#         For integer/None inputs, if ``y`` is binary or multiclass,
#         :class:`StratifiedKFold` used. If the estimator is not a classifier
#         or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
#
#         Refer :ref:`User Guide <cross_validation>` for the various
#         cross-validators that can be used here.
#
#     n_jobs : int or None, default=None
#         Number of jobs to run in parallel.
#         ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
#         ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
#         for more details.
#
#     train_sizes : array-like of shape (n_ticks,)
#         Relative or absolute numbers of training examples that will be used to
#         generate the learning curve. If the ``dtype`` is float, it is regarded
#         as a fraction of the maximum size of the training set (that is
#         determined by the selected validation method), i.e. it has to be within
#         (0, 1]. Otherwise it is interpreted as absolute sizes of the training
#         sets. Note that for classification the number of samples usually have
#         to be big enough to contain at least one sample from each class.
#         (default: np.linspace(0.1, 1.0, 5))
#     """
#     if axes is None:
#         _, axes = plt.subplots(1, 1, figsize=(20, 5))
#
#     axes.set_title(title)
#     if ylim is not None:
#         axes.set_ylim(*ylim)
#     axes.set_xlabel("Training examples")
#     axes.set_ylabel("Score")
#
#     train_sizes, train_scores, test_scores, fit_times, _ = \
#         learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
#                        train_sizes=train_sizes,
#                        return_times=True)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     # fit_times_mean = np.mean(fit_times, axis=1)
#     # fit_times_std = np.std(fit_times, axis=1)
#
#     # Plot learning curve
#     axes.grid()
#     axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                          train_scores_mean + train_scores_std, alpha=0.1,
#                          color="r")
#     axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                          test_scores_mean + test_scores_std, alpha=0.1,
#                          color="g")
#
#     axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
#                  label="Training score")
#     axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
#                  label="Cross-validation score")
#     axes.legend(loc="best")
#
#     # Plot n_samples vs fit_times
#     # axes[1].grid()
#     # axes[1].plot(train_sizes, fit_times_mean, 'o-')
#     # axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
#     #                      fit_times_mean + fit_times_std, alpha=0.1)
#     # axes[1].set_xlabel("Training examples")
#     # axes[1].set_ylabel("fit_times")
#     # axes[1].set_title("Scalability of the model")
#
#     # Plot fit_time vs score
#     # axes[2].grid()
#     # axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
#     # axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
#     #                      test_scores_mean + test_scores_std, alpha=0.1)
#     # axes[2].set_xlabel("fit_times")
#     # axes[2].set_ylabel("Score")
#     # axes[2].set_title("Performance of the model")
#
#     return plt


# fig, axes = plt.subplots(3, 2, figsize=(10, 15))

# X, y = load_digits(return_X_y=True)

# title = "Learning Curves"
#
# estimator = clf
# plot_learning_curve(estimator, title, X, Y, axes=None, ylim=(0.7, 1.01), train_sizes=[0.3, 0.5, 0.6,  0.7,  0.8,  0.9])

# title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# # SVC is more expensive so we do a lower number of CV iterations:
# cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# estimator = SVC(gamma=0.001)
# plot_learning_curve(estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01),
#                     cv=cv, n_jobs=4)
# ds_tree = tree.DecisionTreeClassifier()
# parameters = {'max_depth': list(range(1, 20)),
#               'criterion': ['gini', 'entropy'],
#               'min_samples_split': range(2, 10),
#               'min_samples_leaf': range(2, 10),
#               }
# gs = GridSearchCV(ds_tree, parameters)
# gs.fit(X, Y)
# print(gs.best_params_)

clf = tree.DecisionTreeClassifier(**{'criterion': 'entropy', 'min_samples_leaf': 3, 'min_samples_split': 3})
max_depth = list(range(1, 20))

train_scores, valid_scores = validation_curve(clf, X, Y, "max_depth", max_depth,
                                              cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(valid_scores, axis=1)
# print(train_scores)
# print(valid_scores)
# print(max_depth)
# fig = plt.figure()
# ax = plt.axes()
plt.plot(max_depth, train_scores_mean, label="train_scores")
plt.plot(max_depth, test_scores_mean, label="valid_scores")
plt.legend()
plt.title(f"Validation curve for Decision Tree (Max Depth vs Accuracy)")
plt.savefig('Validation Curve.png')

plt.clf()

clf = tree.DecisionTreeClassifier(**{'criterion': 'entropy', 'min_samples_leaf': 3, 'min_samples_split': 3, 'max_depth': 11})
train_sizes, train_scores, validation_scores = learning_curve(clf, X, Y, cv=5, train_sizes=[0.3, 0.5, 0.7, 0.9, 1])

train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(validation_scores, axis=1)


plt.plot(train_sizes, train_scores_mean, label="train_scores")
plt.plot(train_sizes, val_scores_mean, label="valid_scores")
plt.legend()
plt.title(f"Learning curve for Decision Tree (Training Size vs Accuracy)")
plt.savefig('Learning Curve.png')

clf.fit(X, Y)
predicted_label = clf.predict(car_valuation.test_data)
print(f"Accuracy of test data {accuracy_score(predicted_label, car_valuation.test_label)}")