import numpy as np
import matplotlib.pyplot as plt
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import mean_squared_error

from dataset import car_valuation
from dataset import occupancy
from clustering_em import graph_em_method_clustering_for_data
from clustering_kmean import graph_k_mean_clustering_for_data

MAX_COMPONENTS = 5
comp_range = range(1, MAX_COMPONENTS + 1)


def compute_reconstruction_error(n_components, data):

    random_projection = SparseRandomProjection(n_components=n_components)

    random_projection.fit(data)
    components = random_projection.components_.toarray()
    p_inverse = np.linalg.pinv(components.T)

    reduced_data = random_projection.transform(data)
    reconstructed = reduced_data.dot(p_inverse)
    assert data.shape == reconstructed.shape
    return mean_squared_error(data, reconstructed)


def get_transformed_data_by_random_projection(
        data,
        cluster_range,
        graph=True,
        title=None
):
    reconstruction_error_list = []
    for comp in cluster_range:
        reconstruction_error_list.append(
            compute_reconstruction_error(
                comp,
                data
            )
        )

    if graph and title:
        plt.plot(comp_range, reconstruction_error_list)
        plt.xlabel("Number of components")
        plt.ylabel("Reconstruction Error")
        plt.legend()
        plt.title(title)
        plt.savefig(title)
        plt.clf()

    min_index = np.argmin(reconstruction_error_list)
    random_projection = SparseRandomProjection(n_components=(min_index + 1))
    return random_projection.fit_transform(data), min_index


rdp_transformed_data_car_evaluation, min_index = get_transformed_data_by_random_projection(
    car_valuation.training_data,
    comp_range,
    title="Random Projection for Car Evaluation"
)
rdp_transformed_test_data_car_evaluation = SparseRandomProjection(n_components=(min_index + 1)
                                                                  ).fit_transform(car_valuation.test_data)

graph_k_mean_clustering_for_data(
    data=rdp_transformed_data_car_evaluation,
    cluster_range=comp_range,
    title="K Mean clustering for Car Evaluation Random Projection"
)
graph_em_method_clustering_for_data(
    rdp_transformed_data_car_evaluation,
    "EM clustering for Car Evaluation Random Projection"
)


rdp_transformed_data_occupancy, _ = get_transformed_data_by_random_projection(
    occupancy.training_data,
    comp_range,
    title="Random Projection for Occupancy"
)
graph_k_mean_clustering_for_data(
    data=rdp_transformed_data_occupancy,
    cluster_range=comp_range,
    title="K Mean clustering for Occupancy Random Projection"
)
graph_em_method_clustering_for_data(
    rdp_transformed_data_occupancy,
    "EM clustering for Occupancy Random Projection"
)
