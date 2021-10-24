import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from dataset import car_valuation
from dataset import occupancy

# Choose dataset

# dataset = car_valuation
dataset = occupancy

X = dataset.training_data
Y = dataset.training_label

k_nn = KNeighborsClassifier()
parameters = {
    "n_neighbors": list(range(1, 20)),
}
gs = GridSearchCV(k_nn, parameters)
gs.fit(X, Y)
print(gs.best_params_)

#      Output Car
######################
# {'n_neighbors': 5} #
######################

#      Output Occupancy
######################
# {'n_neighbors': 7} #
######################

val_k_nn = KNeighborsClassifier()
n_neighbors = list(range(1, 20))

train_scores, valid_scores = validation_curve(
    val_k_nn, X, Y, "n_neighbors", n_neighbors, cv=5
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(valid_scores, axis=1)
plt.plot(n_neighbors, train_scores_mean, label="train_scores")
plt.plot(n_neighbors, test_scores_mean, label="valid_scores")
plt.xticks(n_neighbors)
plt.legend()
plt.title(f"Validation curve for K-Neighbors (Number of Neighbors vs Accuracy)")
plt.savefig("Validation Curve.png")

plt.clf()

learning_k_nn = KNeighborsClassifier(n_neighbors=7)
train_size = np.linspace(0.1, 1, 10)
train_sizes, train_scores, validation_scores = learning_curve(
    learning_k_nn, X, Y, cv=5, train_sizes=train_size
)
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(validation_scores, axis=1)


plt.plot(train_sizes, train_scores_mean, label="train_scores")
plt.plot(train_sizes, val_scores_mean, label="valid_scores")
plt.legend()
plt.title(f"Learning curve for K-Neighbors (Number of Neighbors vs Accuracy)")
plt.savefig("Learning Curve.png")

learning_k_nn.fit(X, Y)
predicted_label = learning_k_nn.predict(dataset.test_data)
print(f"Accuracy of test data {accuracy_score(predicted_label, dataset.test_label)}")
