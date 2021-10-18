import mlrose_hiive as mlrose
import numpy as np
import matplotlib.pyplot as plt
import timeit


SEED = 42
MAX_PROBLEM_SIZE = 150
init_state = np.array([i for i in range(MAX_PROBLEM_SIZE)])


fitness = mlrose.fitness.continuous_peaks.ContinuousPeaks()
problem = mlrose.DiscreteOpt(
    length=len(init_state),
    fitness_fn=fitness,
    maximize=True,
)

# Define decay schedule
schedule = mlrose.ExpDecay()


sizes = []
simulated_annealing_best_fitness_list = []
simulated_annealing_run_time_list = []

random_hill_climb_best_fitness_list = []
random_hill_climb_run_time_list = []

genetic_alg_best_fitness_list = []
genetic_alg_run_time_list = []

mimic_best_fitness_list = []
mimic_run_time_list = []

sa_curve, rhc_curve, ga_curve, mimic_curve = None, None, None, None
for size in range(0, MAX_PROBLEM_SIZE + 1, 10):
    if not size:
        continue

    print(f"Problem size {size}")
    init_state = np.array([i for i in range(size)])
    problem = mlrose.DiscreteOpt(
        length=len(init_state),
        fitness_fn=fitness,
        maximize=True,
    )

    start = timeit.default_timer()
    simulated_annealing_best_state, simulated_annealing_best_fitness, sa_curve = mlrose.simulated_annealing(
        problem,
        schedule=schedule,
        max_attempts=10,
        max_iters=200,
        init_state=init_state,
        random_state=SEED,
        curve=True
    )
    stop = timeit.default_timer()
    simulated_annealing_run_time_list.append(stop - start)

    start = timeit.default_timer()
    random_hill_climb_best_state, random_hill_climb_best_fitness, rhc_curve = mlrose.random_hill_climb(
        problem,
        max_attempts=10,
        max_iters=200,
        init_state=init_state,
        random_state=SEED,
        curve=True
    )
    stop = timeit.default_timer()
    random_hill_climb_run_time_list.append(stop - start)

    start = timeit.default_timer()
    genetic_alg_best_state, genetic_alg_best_fitness, ga_curve = mlrose.genetic_alg(
        problem,
        max_attempts=10,
        max_iters=200,
        random_state=SEED,
        curve=True
    )
    stop = timeit.default_timer()
    genetic_alg_run_time_list.append(stop - start)

    start = timeit.default_timer()
    mimic_best_state, mimic_best_fitness, mimic_curve = mlrose.mimic(
        problem,
        max_attempts=10,
        max_iters=200,
        random_state=SEED,
        curve=True
    )
    stop = timeit.default_timer()
    mimic_run_time_list.append(stop - start)

    sizes.append(size)
    simulated_annealing_best_fitness_list.append(simulated_annealing_best_fitness)
    random_hill_climb_best_fitness_list.append(random_hill_climb_best_fitness)
    genetic_alg_best_fitness_list.append(genetic_alg_best_fitness)
    mimic_best_fitness_list.append(mimic_best_fitness)


plt.plot(sizes, simulated_annealing_best_fitness_list, label="sa")
plt.plot(sizes, random_hill_climb_best_fitness_list, label="rhc")
plt.plot(sizes, genetic_alg_best_fitness_list, label="ga")
plt.plot(sizes, mimic_best_fitness_list, label="mimic")
plt.legend()
plt.title(f"Fitness vs Size for Continuous Peaks")
plt.savefig('Fitness Score vs Size Continuous Peaks.png')

plt.clf()

plt.plot(sizes, simulated_annealing_run_time_list, label="sa")
plt.plot(sizes, random_hill_climb_run_time_list, label="rhc")
plt.plot(sizes, genetic_alg_run_time_list, label="ga")
plt.plot(sizes, mimic_run_time_list, label="mimic")
plt.legend()
plt.title(f"Runtime vs Size for Continuous Peaks")
plt.savefig('Runtime vs Size for Continuous Peaks.png')

plt.clf()

plt.plot(range(len(sa_curve[:, 1])), np.cumsum(sa_curve[:, 1]), label="sa")
plt.plot(range(len(rhc_curve[:, 1])), np.cumsum(rhc_curve[:, 1]), label="rhc")
plt.plot(range(len(ga_curve[:, 1])), np.cumsum(ga_curve[:, 1]), label="ga")
plt.plot(range(len(mimic_curve[:, 1])), np.cumsum(mimic_curve[:, 1]), label="mimic")
plt.legend()
plt.title(f"Function Evaluation Continuous Peaks (Cumulative Func Evaluation vs Iters)")
plt.savefig('Function Evaluation Continuous Peaks')

plt.clf()


# temp = []
# simulated_annealing_best_fitness_list = []
# for t in range(0, 1000, 10):
#     if not t:
#         continue
#
#     schedule = mlrose.ExpDecay(init_temp=t)
#     init_state = np.array([0 for i in range(MAX_PROBLEM_SIZE)])
#     problem = mlrose.DiscreteOpt(
#         length=len(init_state),
#         fitness_fn=fitness,
#         maximize=True,
#     )
#     simulated_annealing_best_state, simulated_annealing_best_fitness, _ = mlrose.simulated_annealing(
#         problem,
#         schedule=schedule,
#         max_attempts=10,
#         max_iters=200,
#         init_state=init_state,
#         random_state=SEED,
#         curve=True
#     )
#     simulated_annealing_best_fitness_list.append(simulated_annealing_best_fitness)
#     temp.append(t)
#
# plt.plot(temp, simulated_annealing_best_fitness_list)
# plt.title(f"Best Fitness Score vs Initial Temperature for Continuous Peaks")
# plt.savefig('Best Fitness Score vs Initial Temperature for Continuous Peaks.png')
