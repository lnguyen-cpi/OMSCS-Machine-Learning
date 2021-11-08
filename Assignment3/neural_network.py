import matplotlib.pyplot as plt
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from pca import pca_transformed_data_car_valuation, pca_transformed_test_data_car_valuation
from ica import ica_transformed_data_car_evaluation, ica_transformed_test_data_car_evaluation
from randomize_proj import rdp_transformed_data_car_evaluation, rdp_transformed_test_data_car_evaluation

from dataset import car_valuation


def graph_neural_network_performance_for_car_valuation_dataset(
        data,
        labels,
        test_data=None,
        test_label=None,
        seed=42,
        title=None

):
    max_iter = 1000
    learning_neural_network = MLPClassifier(
        random_state=seed,
        hidden_layer_sizes=(6,) * 10,
        max_iter=max_iter
    )
    learning_neural_network.fit(
        data,
        labels
    )
    if title:
        plt.plot(learning_neural_network.loss_curve_)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title(title)
        plt.savefig(title)
        plt.clf()

    if test_data is not None and test_label is not None:
        pred_labels = learning_neural_network.predict(test_data)
        print(f"Accuracy for Neural network is {accuracy_score(pred_labels, test_label)}")


graph_neural_network_performance_for_car_valuation_dataset(
    pca_transformed_data_car_valuation,
    car_valuation.training_label,
    pca_transformed_test_data_car_valuation,
    car_valuation.test_label,
    title="Loss Curve for reduced Car Evaluation PCA"
)
graph_neural_network_performance_for_car_valuation_dataset(
    ica_transformed_data_car_evaluation,
    car_valuation.training_label,
    ica_transformed_test_data_car_evaluation,
    car_valuation.test_label,
    title="Loss Curve for reduced Car Evaluation ICA"
)
graph_neural_network_performance_for_car_valuation_dataset(
    rdp_transformed_data_car_evaluation,
    car_valuation.training_label,
    rdp_transformed_test_data_car_evaluation,
    car_valuation.test_label,
    title="Loss Curve for reduced Car Evaluation Random Projection"
)

graph_neural_network_performance_for_car_valuation_dataset = graph_neural_network_performance_for_car_valuation_dataset

