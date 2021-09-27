import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from dataset import car_valuation
from dataset import occupancy

# dataset = car_valuation
dataset = occupancy

X = dataset.training_data
Y = dataset.training_label

linear_svm_classifier = SVC(kernel='linear', random_state=42)


# gs = GridSearchCV(linear_svm_classifier,
#                   {
#                       'C': list(range(1, 7))
#                   }
#                   )
# gs.fit(X, Y)
# print(gs.best_params_)


c_range = list(range(1, 7))
train_scores, valid_scores = validation_curve(linear_svm_classifier, X, Y, "C", c_range, cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(valid_scores, axis=1)
plt.plot(c_range, train_scores_mean, label="train_scores")
plt.plot(c_range, test_scores_mean, label="valid_scores")
plt.legend()
plt.title(f"Validation curve for LinearSVM (C vs Accuracy)")
plt.savefig('Validation Curve LinearSVM.png')

plt.clf()

# Output Car valuation
# C = 2.

# Output Occupancy
# C = 1.

# poly_svm_classifier = SVC(kernel='poly', random_state=42)
# gs = GridSearchCV(poly_svm_classifier,
#                   {
#                       'degree': list(range(10)),
#                       'C': list(range(1, 5)),
#                       'gamma': ['scale', 'auto']
#                   }
#                   )
# gs.fit(X, Y)
# print(gs.best_params_)


# Output Car valuation
# # {'C': 1, 'degree': 5, 'gamma': 'scale'}

# Output Occupancy
# # {'C': 7, 'degree': 4, 'gamma': 'scale'}


poly_svm_classifier = SVC(kernel='poly', C=1, gamma='scale', random_state=42)
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


# best_svm_classifier = SVC(kernel='poly', C=1, gamma='scale', degree=5, random_state=42)
best_svm_classifier = SVC(kernel='linear', random_state=42, C=1)


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
predicted_label = best_svm_classifier.predict(dataset.test_data)
print(f"Accuracy of test data {accuracy_score(predicted_label, dataset.test_label)}")

# Accuracy of test data 0.94
