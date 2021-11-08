import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from dataset import car_valuation
from dataset import occupancy
from clustering_em import graph_em_method_clustering_for_data
from clustering_kmean import graph_k_mean_clustering_for_data
from pca import pca_transformed_data_car_valuation, pca_transformed_test_data_car_valuation

from ica import ica_transformed_data_car_evaluation, ica_transformed_test_data_car_evaluation
from randomize_proj import rdp_transformed_data_car_evaluation, rdp_transformed_test_data_car_evaluation
from neural_network import graph_neural_network_performance_for_car_valuation_dataset

MAX_COMPONENTS = 6
cluster_range = range(1, MAX_COMPONENTS + 1)


def transform_k_mean_cluster(
        data,
        n_clusters,
        seed=42
):
    model = KMeans(n_clusters=n_clusters,random_state=seed)
    model.fit(data)
    cluster_labels = model.predict(data)
    cluster_labels = np.reshape(cluster_labels, (len(cluster_labels), 1))
    return np.append(data, cluster_labels, 1)


def transform_em_cluster(
        data,
        n_clusters,
        seed=42
):
    scaler = preprocessing.StandardScaler().fit(data)
    scaled_dataset = scaler.transform(data)
    gmm = GaussianMixture(n_components=n_clusters, random_state=seed)
    gmm.fit(scaled_dataset)
    cluster_labels = gmm.predict(data)
    cluster_labels = np.reshape(cluster_labels, (len(cluster_labels), 1))
    return np.append(data, cluster_labels, 1)


k_mean_pca_transformed = transform_k_mean_cluster(
    pca_transformed_data_car_valuation,
    3
)
k_mean_test_pca_transformed = transform_k_mean_cluster(
    pca_transformed_test_data_car_valuation,
    3
)
em_pca_transformed = transform_em_cluster(
    pca_transformed_data_car_valuation,
    6
)


graph_neural_network_performance_for_car_valuation_dataset(
    k_mean_pca_transformed,
    car_valuation.training_label,
    k_mean_test_pca_transformed,
    car_valuation.training_label,
    title="K-Mean cluster for PCA Car Evaluation Loss curve"
)

graph_neural_network_performance_for_car_valuation_dataset(
    em_pca_transformed,
    car_valuation.training_label,
    title="EM cluster for PCA Car Evaluation Loss curve"
)



k_mean_ica_transformed = transform_k_mean_cluster(
    ica_transformed_data_car_evaluation,
    3
)
k_mean_test_ica_transformed = transform_k_mean_cluster(
    ica_transformed_data_car_evaluation,
    3
)
em_ica_transformed = transform_em_cluster(
    ica_transformed_data_car_evaluation,
    6
)

graph_neural_network_performance_for_car_valuation_dataset(
    k_mean_ica_transformed,
    car_valuation.training_label,
    k_mean_test_ica_transformed,
    car_valuation.test_label,
    title="K-Mean cluster for ICA Car Evaluation Loss curve"
)

graph_neural_network_performance_for_car_valuation_dataset(
    em_ica_transformed,
    car_valuation.training_label,
    title="EM cluster for ICA Car Evaluation Loss curve"
)



k_mean_rand_proj_transformed = transform_k_mean_cluster(
    ica_transformed_data_car_evaluation,
    4
)
k_mean_test_rand_proj_transformed = transform_k_mean_cluster(
    ica_transformed_data_car_evaluation,
    4
)
em_rand_proj_transformed = transform_em_cluster(
    ica_transformed_data_car_evaluation,
    6
)

graph_neural_network_performance_for_car_valuation_dataset(
    k_mean_rand_proj_transformed,
    car_valuation.training_label,
    k_mean_test_rand_proj_transformed,
    car_valuation.test_label,
    title="K-Mean cluster for Random Projection Car Evaluation Loss curve"
)

graph_neural_network_performance_for_car_valuation_dataset(
    em_rand_proj_transformed,
    car_valuation.training_label,
    title="EM cluster for Random Projection Car Evaluation Loss curve"
)
