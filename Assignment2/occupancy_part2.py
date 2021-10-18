import mlrose_hiive as mlrose
import numpy as np
import matplotlib.pyplot as plt
import timeit

from sklearn.metrics import accuracy_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import learning_curve

from dataset import occupancy

dataset = occupancy


def get_random_hill_climb_model():

    return mlrose.NeuralNetwork(
        hidden_nodes=[6] * 10,
        activation='relu',
        algorithm='random_hill_climb',
        max_iters=1000,
        is_classifier=True,
        learning_rate=0.001,
        random_state=42,
        curve=True
    )


def get_simulated_annealing_model():
    return mlrose.NeuralNetwork(
        hidden_nodes=[6] * 10,
        activation='relu',
        algorithm='simulated_annealing',
        max_iters=1000,
        is_classifier=True,
        learning_rate=0.001,
        random_state=42,
        curve=True,
        schedule=mlrose.GeomDecay(init_temp=100)
    )


def get_genetic_alg_model():
    return mlrose.NeuralNetwork(
        hidden_nodes=[6] * 10,
        activation='relu',
        algorithm='genetic_alg',
        max_iters=1000,
        is_classifier=True,
        learning_rate=0.001,
        random_state=42,
        curve=True
    )


train_size = np.linspace(0.1, 1, 10)
N = len(dataset.training_data)
# One hot encode target values
one_hot = OneHotEncoder()
sizes = []
rhc_accuracy = []
rhc_runtime = []
sa_accuracy = []
sa_runtime = []
ga_accuracy = []
ga_runtime = []
for size in train_size:
    print(f"size {size}")
    sizes.append(size)
    random_hill_climb_model = get_random_hill_climb_model()
    sa_model = get_simulated_annealing_model()
    ga_model = get_genetic_alg_model()
    training_data = dataset.training_data[:int(size * N)]
    training_label = dataset.training_label[:int(size * N)]
    y_train_hot = one_hot.fit_transform(training_label).todense()

    start = timeit.default_timer()
    random_hill_climb_model.fit(training_data, y_train_hot)
    stop = timeit.default_timer()
    rhc_runtime.append(stop - start)

    start = timeit.default_timer()
    sa_model.fit(training_data, y_train_hot)
    stop = timeit.default_timer()
    sa_runtime.append(stop - start)

    start = timeit.default_timer()
    ga_model.fit(training_data, y_train_hot)
    stop = timeit.default_timer()
    ga_runtime.append(stop - start)

    rhc_pred_label = random_hill_climb_model.predict(training_data)
    ga_pred_label = ga_model.predict(training_data)
    sa_pred_label = sa_model.predict(training_data)

    rhc_accuracy.append(accuracy_score(y_train_hot, rhc_pred_label))
    sa_accuracy.append(accuracy_score(y_train_hot, sa_pred_label))
    ga_accuracy.append(accuracy_score(y_train_hot, ga_pred_label))


plt.plot(sizes, rhc_accuracy, label="rhc")
plt.plot(sizes, sa_accuracy, label="sa")
plt.plot(sizes, ga_accuracy, label="ga")
plt.legend()
plt.title(f"Learning curve for Occupancy (Accuracy vs Training Size)")
plt.savefig('Learning curve for Occupancy.png')

plt.clf()


plt.plot(sizes, rhc_runtime, label="rhc")
plt.plot(sizes, sa_runtime, label="sa")
plt.plot(sizes, ga_runtime, label="ga")
plt.legend()
plt.title(f"Run time for Occupancy (Accuracy vs Training Size)")
plt.savefig('Run time for Occupancy.png')

plt.clf()

# Get test accuracy
y_train_hot = one_hot.fit_transform(dataset.training_label).todense()
y_test_hot = one_hot.fit_transform(dataset.test_label).todense()
rhc_model = get_random_hill_climb_model()
sa_model = get_simulated_annealing_model()
ga_model = get_genetic_alg_model()

rhc_model.fit(dataset.training_data, y_train_hot)
sa_model.fit(dataset.training_data, y_train_hot)
ga_model.fit(dataset.training_data, y_train_hot)

rhc_pred_label = rhc_model.predict(dataset.test_data)
ga_pred_label = ga_model.predict(dataset.test_data)
sa_pred_label = sa_model.predict(dataset.test_data)

print(f"Random Hill Climbing Test Accuracy: {accuracy_score(y_test_hot, rhc_pred_label)}")
print(f"Simulated Annealing Test Accuracy: {accuracy_score(y_test_hot, sa_pred_label)}")
print(f"Genetic Algorithm Test Accuracy: {accuracy_score(y_test_hot, ga_pred_label)}")
