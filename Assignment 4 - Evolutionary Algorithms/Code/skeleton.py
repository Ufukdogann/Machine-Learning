#
# USI - Universita della svizzera italiana
#
# Machine Learning - Fall 2018
#
# Assignment 5: Evolutionary Algorithms
# 
# (!) This code skeleton is just a recommendation, you don't have to use it.
#

import numpy as np
import matplotlib.pyplot as plt
import itertools
import random


def get_fitness(chromosome):

	target = 1
	fitness_score = 0
	for k, g in itertools.groupby(chromosome):
		if k == target and len(list(g)) > 1:
			fitness_score += 1

	return fitness_score


def fitness_proportional_selection(fitnesses):

	best_chromosome = np.argsort(fitnesses)[::-1][0]
	return best_chromosome


def bitflip_mutatation(chromosome, mutation_rate):

	for i in chromosome:
		result = np.random.rand(1)[0]
		if result <= mutation_rate:
			if chromosome[i] == 1:
				chromosome[i] = 0
			else:
				chromosome[i] = 1
	return chromosome


def two_point_crossover(parentA, parentB):

	for i in range(1):
		crossover_position_one = random.randint(0, len(parentA) - 1)
		crossover_position_two = random.randint(0, len(parentB) - 1)
		if crossover_position_one>crossover_position_two:
			temp = crossover_position_two
			crossover_position_one = temp
			crossover_position_two = crossover_position_one

	first_time_splitted_first_chromosome = list(parentA[:crossover_position_one])
	second_time_splitted_first_chromosome = list(parentA[crossover_position_one:crossover_position_two])
	rest_of_first_chromosome = list(parentA[crossover_position_two:])

	first_time_splitted_second_chromosome = list(parentB[:crossover_position_one])
	second_time_splitted_second_chromosome = list(parentB[crossover_position_one:crossover_position_two])
	rest_of_second_chromosome = list(parentB[crossover_position_two:])

	child_A = (
				first_time_splitted_first_chromosome + second_time_splitted_second_chromosome + rest_of_first_chromosome)
	child_B = (
				second_time_splitted_first_chromosome + first_time_splitted_second_chromosome + rest_of_second_chromosome)

	return child_A, child_B


def generate_initial_population(length, population_size):
	initial_population = []
	for i in range(population_size):
		chromosome = np.random.randint(2, size=length)
		initial_population.append(chromosome)
	return initial_population


def ga(length, population_size, mutation_rate, cross_over_rate=1.0, max_gen=100):
	population = generate_initial_population(length, population_size)
	print(population)
	best_chromosome = []
	for i in range(max_gen):
		mutated_children = []
		fitness_score = [get_fitness(chromosome) for chromosome in population]
		best_fitness_index = np.argmax(fitness_score)
		best_fitness = population[best_fitness_index]

		for j in range(population_size//2):
			population_copy = list(np.copy(population))
			first_chromosome_index = fitness_proportional_selection(fitness_score)
			parent_A = population_copy[first_chromosome_index]
			del fitness_score[first_chromosome_index]
			second_chromosome_index = fitness_proportional_selection(fitness_score)
			parent_B = population_copy[second_chromosome_index]
			del fitness_score[second_chromosome_index]

			child_A, child_B = two_point_crossover(parentA=parent_A, parentB=parent_B)

			mutated_child_A = bitflip_mutatation(child_A, mutation_rate)
			mutated_child_B = bitflip_mutatation(child_B, mutation_rate)

			mutated_children.append(mutated_child_A)
			mutated_children.append(mutated_child_B)

		best_chromosome.append(best_fitness)
		population = mutated_children
	return population,  best_chromosome


def plot_minmax_curve(run_stats):
	min_length = min(len(r) for r in run_stats)
	truncated_stats = np.array([r[:min_length] for r in run_stats])

	X = np.arange(truncated_stats.shape[1])
	means = truncated_stats.mean(axis=0)
	mins = truncated_stats.min(axis=0)
	maxs = truncated_stats.max(axis=0)

	fig, ax = plt.subplots(figsize=(12, 6))
	ax.plot(means, '-o')

	ax.fill_between(X, mins[:, 0], maxs[:, 0], linewidth=0, facecolor="b", alpha=0.3, interpolate=True)
	ax.fill_between(X, mins[:, 1], maxs[:, 1], linewidth=0, facecolor="g", alpha=0.3, interpolate=True)
	ax.fill_between(X, mins[:, 2], maxs[:, 2], linewidth=0, facecolor="r", alpha=0.3, interpolate=True)


def run_part1(length=5):
	run_stats = []
	for run in range(10):
		run_stat, _ = ga(length=length, population_size=40, mutation_rate=0.02, max_gen=60)
		run_stats.append(run_stat)
	plot_minmax_curve(run_stats)

N= [5, 10, 20, 50]
for a in N:
	average = 0
	for i in range(10):
		best_fitness_array = []
		final, best_chromosome = ga(a, 40, 0.2, 2, 100)
		fitness_score = [get_fitness(chromosome) for chromosome in final]
		best_fitness_index = np.argmax(fitness_score)
		best_fitness = final[best_fitness_index]
		best_fitness_score = get_fitness(best_fitness)
		for j in range(len(final)):
			print(final[j])
		print("Best Chromosome:" + str(best_fitness))
		print("Best Fitness Score:" + str(best_fitness_score))
		fitness_score = [get_fitness(chromosome) for chromosome in best_chromosome]
		print("Fitness Scores:" + str(fitness_score))
		best_fitness_array.append(fitness_score)
	print(best_fitness_array)
	numpy_array = np.array(best_fitness_array)
	averages = np.mean(numpy_array, axis=0)
	plt.plot(range(1, len(averages)+1), averages, label="N= " + str(a))
	plt.legend(loc='best')
plt.show()



# part 2
def rosenbrock(x):
	return np.sum((1-x[:-1])**2 + 100*(x[1:] - x[:-1]**2)**2, axis=0)


def plot_surface():
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm

	G = np.meshgrid(np.arange(-1.0, 1.5, 0.05), np.arange(-1.0, 1.5, 0.05))
	R = rosenbrock(np.array(G))

	fig = plt.figure(figsize=(14,9))
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(G[0], G[1], R.T, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
	ax.set_zlim(0.0, 500.0)
	ax.view_init(elev=50., azim=230)

	plt.show()


def sample_offspring(parents, lambda_, tau, tau_prime, epsilon0=0.001):
	pass


def ES(N=5, mu=2, lambda_=100, generations=100, epsilon0=0.001):

	total_number_of_parents = mu
	parents_array = []
	sigmoid_array = []
	tau = 1/np.sqrt(N)
	epsilon_i = np.zeros(N)
	teta_prime_array = np.zeros(N)
	all_teta_primes = []
	all_sigma_primes = []
	fitnesses = []
	list_of_mean_smallest_fitness, list_of_variance = [], []


	for i in range(total_number_of_parents):
		parent = np.random.uniform(-5, 10, N)
		parents_array.append(parent)
		sigmoid = np.random.uniform(0, 1, N)
		sigmoid_array.append(sigmoid)

	numpy_parents_array = np.array(parents_array)
	numpy_sigmoid_array = np.array(sigmoid_array)

	for g in range(generations):

		for i in range(lambda_):
			averages_of_parents = np.mean(numpy_parents_array, axis=0)
			averages_of_sigmoid = np.mean(numpy_sigmoid_array, axis=0)
			first_epsilon = np.random.normal(0, tau, 1)[0]
			sigma_prime = np.exp(first_epsilon) * averages_of_sigmoid
			for j in range(N):
				if sigma_prime[j] < epsilon0:
					sigma_prime[j] = epsilon0
				epsilon_i[j] = np.random.normal(0, sigma_prime[j], 1)[0]
			teta_prime_array = averages_of_parents + epsilon_i
			all_teta_primes.append(teta_prime_array)
			all_sigma_primes.append(sigma_prime)

		numpy_all_teta_primes = np.array(all_teta_primes)
		numpy_all_sigmoid_primes = np.array(all_sigma_primes)
		fitnesses = []
		for i in range(len(all_teta_primes)):
			fitness = rosenbrock(all_teta_primes[i])
			fitnesses.append(fitness)

		smallest_fitness = np.argsort(fitnesses)[:mu]
		numpy_parents_array = numpy_all_teta_primes[smallest_fitness]
		numpy_sigmoid_array = numpy_all_sigmoid_primes[smallest_fitness]
		numpy_fitnesses_array = np.array(fitnesses)
		smallest_fitness_array = numpy_fitnesses_array[smallest_fitness]

		mean_smallest_fitness = np.mean(smallest_fitness_array)
		variance = np.var(smallest_fitness_array)
		list_of_mean_smallest_fitness.append(mean_smallest_fitness)
		list_of_variance.append(variance)

	return list_of_variance, list_of_mean_smallest_fitness



def plot_ES_curve(F):
	min_length = min(len(f) for f in F)
	F_plot = np.array([f[:min_length] for f in F])
	fig, ax = plt.subplots(figsize=(12, 6))
	ax.plot(np.mean(F_plot.T, axis=1))
	ax.fill_between(range(min_length), np.min(F_plot.T, axis=1), np.max(F_plot.T, axis=1), linewidth=0, facecolor="b", alpha=0.3, interpolate=True)
	ax.set_yscale('log')


def run_part2(length=5):
	run_stats = []
	for i in range(10):
		fit, solution = ES(N=length, mu=10, lambda_=100, epsilon0=0.0001, generations=500)
		run_stats.append(fit)
	plot_ES_curve(run_stats)

N = [5, 10, 20, 50]
col = ['k','y','r','m']
final_variance_list, final_smallest_fitness = [], []

for i in range(len(N)):
	for a in range(10):
		list_of_variance, list_of_mean_smallest_fitness = ES(N[i], 10, 100, 100, 0.0001)
	final_smallest_fitness.append(list_of_mean_smallest_fitness)
	final_variance_list.append(list_of_variance)
	numpy_final_smallest_fitness = np.array(final_smallest_fitness)
	numpy_final_variance = np.array(final_variance_list)
	mean_of_numpy_final_smallest_fitness = np.mean(numpy_final_smallest_fitness, axis=0)
	mean_numpy_final_variance = np.mean(numpy_final_variance, axis=0)
	plt.plot(range(1, len(mean_of_numpy_final_smallest_fitness)+1), mean_of_numpy_final_smallest_fitness, c=col[i], label="smallest_fitness" + str(N[i]))
	plt.plot(range(1, len(mean_numpy_final_variance) + 1), mean_numpy_final_variance, c=col[i], linestyle='--', label="variance" + str(N[i]))
	plt.grid()
	plt.yscale('log')
	plt.legend(loc='best', fontsize='medium')
plt.show()



