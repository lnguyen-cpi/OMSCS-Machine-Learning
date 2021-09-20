import math
import numpy as np
from sklearn import tree
from dataset import car_valuation
from sklearn.model_selection import learning_curve


from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

X = car_valuation.training_data
Y = car_valuation.training_label

linear_svm_classifier = SVC(kernel='linear', random_state=42)


# gs = GridSearchCV(linear_svm_classifier,
#                   {
#                       'C': list(range(1, 101))
#                   }
#                   )
# gs.fit(X, Y)
# print(gs.best_params_)
#

c_range = list(range(1, 101))
train_scores, valid_scores = validation_curve(linear_svm_classifier, X, Y, "C", c_range, cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(valid_scores, axis=1)
plt.plot(c_range, train_scores_mean, label="train_scores")
plt.plot(c_range, test_scores_mean, label="valid_scores")
plt.legend()
plt.title(f"Validation curve for LinearSVM (C vs Accuracy)")
plt.savefig('Validation Curve LinearSVM.png')

plt.clf()

# C = 2.

# poly_svm_classifier = SVC(kernel='poly', random_state=42)
# gs = GridSearchCV(poly_svm_classifier,
#                   {
#                       'degree': list(range(8)),
#                       'C': list(range(1, 101)),
#                       'gamma': ['scale', 'auto']
#                   }
#                   )
# gs.fit(X, Y)
# print(gs.best_params_)

# {'C': 7, 'degree': 4, 'gamma': 'scale'}

poly_svm_classifier = SVC(kernel='poly', C=7, gamma='scale', random_state=42)
poly_degree = list(range(8))
train_scores, valid_scores = validation_curve(poly_svm_classifier, X, Y, "degree", poly_degree, cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(valid_scores, axis=1)
plt.plot(poly_degree, train_scores_mean, label="train_scores")
plt.plot(poly_degree, test_scores_mean, label="valid_scores")
plt.legend()
plt.title(f"Validation curve for Poly SVM (Poly Degree vs Accuracy)")
plt.savefig('Validation Curve Poly.png')

plt.clf()


best_svm_classifier = SVC(kernel='poly', C=7, gamma='scale', degree=4, random_state=42)
train_size = np.linspace(0.1, 1, 10)
max_iter = list(list(range(100, 3000, 100)))
train_scores, validation_scores = validation_curve(best_svm_classifier, X, Y, 'max_iter', max_iter, cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(validation_scores, axis=1)


plt.plot(max_iter, train_scores_mean, label="train_scores")
plt.plot(max_iter, val_scores_mean, label="valid_scores")
plt.legend()
plt.title(f"Learning curve for Poly SVM (Max Iteration vs Accuracy)")
plt.savefig('Learning Curve.png')

best_svm_classifier.fit(X, Y)
predicted_label = best_svm_classifier.predict(car_valuation.test_data)
print(f"Accuracy of test data {accuracy_score(predicted_label, car_valuation.test_label)}")

# Accuracy of test data 0.9367283950617284
