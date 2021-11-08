from sklearn.cluster import KMeans
from dataset import car_valuation
from dataset import occupancy
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt

SEED = 42


def graph_k_mean_clustering_for_data(
        data,
        cluster_range,
        title=None,
        seed=SEED,
):
    model = KMeans(random_state=seed)
    elbow_visualizer = KElbowVisualizer(
        model,
        k=cluster_range,
        timings=False,
    )
    elbow_visualizer.fit(data)
    if title:
        elbow_visualizer.set_title(title=title)
        elbow_visualizer.show(
            outpath=title,
            clear_figure=True,
        )

        plt.clf()


graph_k_mean_clustering_for_data(
    data=car_valuation.training_data,
    cluster_range=range(2, 10),
    title="Elbow Clustering Analysis for Car Valuation Dataset"
)
graph_k_mean_clustering_for_data(
    data=occupancy.training_data,
    cluster_range=range(2, 10),
    title="Elbow Clustering Analysis for Occupancy Dataset"
)


