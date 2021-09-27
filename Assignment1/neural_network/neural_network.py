import matplotlib.pyplot as plt
import numpy as np

from sklearn.neural_network import MLPClassifier
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


val_neural_network = MLPClassifier(random_state=42)
hidden_sizes = range(1, 15)
hidden_layer_sizes = [(6, ) * i for i in hidden_sizes]
length_of_hidden_layers = [len(hl) for hl in hidden_layer_sizes]

train_scores, valid_scores = validation_curve(val_neural_network, X, Y, "hidden_layer_sizes", hidden_layer_sizes, cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(valid_scores, axis=1)
plt.plot(length_of_hidden_layers, train_scores_mean, label="train_scores")
plt.plot(length_of_hidden_layers, test_scores_mean, label="valid_scores")
plt.xticks(list(hidden_sizes))
plt.legend()
plt.title(f"Validation curve for NeuralNetwork (Num of Hidden Layers vs Accuracy)")
plt.savefig('Validation Curve.png')

plt.clf()

# Output of Car valuation
########################
# {'hidden_layer_sizes': 6} #
########################


# Output of Occupancy
########################
# {'hidden_layer_sizes': 10} #
########################

max_iter = 1000
learning_neural_network = MLPClassifier(random_state=42, hidden_layer_sizes=(6, ) * 10, max_iter=max_iter)
learning_neural_network.fit(X, Y)


plt.plot(learning_neural_network.loss_curve_)
plt.title(f"Learning curve for NeuralNetwork (Iteration vs Loss)")
plt.savefig('Learning Curve.png')




predicted_label = learning_neural_network.predict(dataset.test_data)
print(f"Accuracy of test data {accuracy_score(predicted_label, dataset.test_label)}")
