import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn import preprocessing
from scipy.stats import kurtosis

from dataset import car_valuation
from dataset import occupancy
from clustering_em import graph_em_method_clustering_for_data
from clustering_kmean import graph_k_mean_clustering_for_data


MAX_COMPONENTS = 5
comp_range = range(1, MAX_COMPONENTS + 1)


def get_transformed_data_by_ica(
        data,
        components,
        graph=True,
        title=None
):
    scaler = preprocessing.StandardScaler().fit(data)
    X_scaled = scaler.transform(data)
    kurtosis_list = []
    for comp in components:
        ica = FastICA(n_components=comp, whiten=True, random_state=42)
        transform_comp = ica.fit_transform(X_scaled)
        kurtosis_list.append(kurtosis(np.average(transform_comp, axis=0)))

    if graph and title:
        plt.plot(components, kurtosis_list)
        plt.xlabel("Number of Components")
        plt.ylabel("Kurtosis")
        plt.legend()
        plt.title(title)
        plt.savefig(title)
        plt.clf()

    max_index = np.argmax(kurtosis_list)
    ica = FastICA(n_components=(max_index + 1), whiten=True, random_state=42)
    return ica.fit_transform(X_scaled)


ica_transformed_data_car_evaluation = get_transformed_data_by_ica(
    car_valuation.training_data,
    comp_range,
    title="ICA for Car Evaluation"
)
scaler = preprocessing.StandardScaler().fit(car_valuation.training_data)
X_scaled = scaler.transform(car_valuation.test_data)
ica_transformed_test_data_car_evaluation = FastICA(n_components=3, whiten=True, random_state=42).fit_transform(X_scaled)

graph_k_mean_clustering_for_data(
    data=ica_transformed_data_car_evaluation,
    cluster_range=comp_range,
    title="K-Mean clustering for Car Evaluation ICA"
)
graph_em_method_clustering_for_data(
    ica_transformed_data_car_evaluation,
    "EM clustering for Car Evaluation ICA"
)


ica_transformed_data_occupancy = get_transformed_data_by_ica(
    occupancy.training_data,
    comp_range,
    title="ICA for Occupancy"
)
graph_k_mean_clustering_for_data(
    data=ica_transformed_data_occupancy,
    cluster_range=comp_range,
    title="K-Mean clustering for Occupancy ICA"
)
graph_em_method_clustering_for_data(
    ica_transformed_data_occupancy,
    "EM clustering for Occupancy ICA"
)

