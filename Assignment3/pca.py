import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dataset import car_valuation
from dataset import occupancy
from clustering_em import graph_em_method_clustering_for_data
from clustering_kmean import graph_k_mean_clustering_for_data

MAX_COMPONENTS = 6
cluster_range = range(1, MAX_COMPONENTS + 1)


def get_transformed_data_by_pca(
        data,
        components=MAX_COMPONENTS,
        graph=True,
        title=None
):
    pca = PCA(n_components=components, svd_solver='full')
    pca.fit(data)
    if graph and title:
        plt.plot(range(1, components + 1), pca.singular_values_)
        plt.title(title)
        plt.xlabel("Number of Components")
        plt.ylabel("Singular values")
        plt.legend()
        plt.savefig(title)
        plt.clf()

    return pca.transform(data)


pca_transformed_data_car_valuation = get_transformed_data_by_pca(
    car_valuation.training_data,
    title="PCA for Car Evaluation"
)
pca_transformed_test_data_car_valuation = get_transformed_data_by_pca(
    car_valuation.test_data,
)
graph_k_mean_clustering_for_data(
    data=pca_transformed_data_car_valuation,
    cluster_range=cluster_range,
    title="K-Mean clustering for Car Evaluation PCA"
)
graph_em_method_clustering_for_data(
    pca_transformed_data_car_valuation,
    "EM clustering for Car Evaluation PCA"
)


pca_transformed_data_occupancy = get_transformed_data_by_pca(
    occupancy.training_data,
    title="PCA for Occupancy"
)
graph_k_mean_clustering_for_data(
    data=pca_transformed_data_occupancy,
    cluster_range=cluster_range,
    title="K-Mean clustering for Occupancy PCA"
)
graph_em_method_clustering_for_data(
    pca_transformed_data_occupancy,
    "EM clustering for Occupancy PCA"
)
