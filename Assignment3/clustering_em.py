from dataset import car_valuation
from dataset import occupancy
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
import matplotlib.pyplot as plt


SEED = 42
cluster_range = range(2, 15)


def graph_em_method_clustering_for_data(
        dataset,
        title=None,
        seed=SEED,
):
    bic_list, aic_list = [], []
    scaler = preprocessing.StandardScaler().fit(dataset)
    scaled_dataset = scaler.transform(dataset)
    for cluster in cluster_range:
        gmm = GaussianMixture(n_components=cluster, random_state=seed)
        gmm.fit(scaled_dataset)
        bic_list.append(gmm.bic(scaled_dataset))
        aic_list.append(gmm.aic(scaled_dataset))

    plt.plot(cluster_range, bic_list, label="bic")
    plt.plot(cluster_range, aic_list, label="aic")
    plt.xlabel("Number of Components")
    plt.legend()
    if title:
        plt.title(title)
        plt.savefig(title)
    plt.clf()


graph_em_method_clustering_for_data(
    car_valuation.training_data,
    "EM clustering for Car Evaluation"
)
graph_em_method_clustering_for_data(
    occupancy.training_data,
    "EM clustering for Occupancy"
)

