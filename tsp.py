import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt


# a function used to score our population
def calculate_fitness(dataset_tuple, population, pop_size, generation):
    # create the fitness array
    fitness_array = np.zeros(shape=pop_size)
    # loop each candidate
    for i in range(0, population.shape[0]): # loop each candidate
        fitness = 0
        print('calculating at generation...' + str(generation) + " for candidate..." + str(i))
        # calculate the rolling difference
        for j in range(0, population.shape[1] - 1):
            # get the distances of city one to city 2 from the population matrix
            distance_a = dataset_tuple[int(population[i][j])][3]
            distance_b = dataset_tuple[int(population[i][j+1])][3]
            distance = abs(distance_a - distance_b)
            fitness = fitness + distance
        fitness_array[i-1] = fitness
    return fitness_array


# a function to compute the euclidean distance between two sets of points
def euclidean_distance(distance_a, distance_b):
    # for speed purposes do not sqrt
    #np.sqrt((distance_a - distance_b) ** 2)
    return (distance_a - distance_b)


# create a population of lists and matricies
def create_population(pop_size, bits):
    population = []
    for i in range(0, pop_size):
        template_population = np.arange(0, (bits))
        np.random.shuffle(template_population)
        index_of_zero = np.where(template_population == 0)[0][0]
        value_at_index_zero = template_population[0]
        template_population[0] = 0
        template_population[index_of_zero] = value_at_index_zero
        population.append(template_population)
        template_population[bits-1] = 0
    # create a 2d matrix from the list of arrays
    return np.array(population)


# a method for mutating the candidates
def mutation(selected_candidate):
    point_a = np.random.randint(1, selected_candidate.size-1)
    point_b = np.random.randint(1, selected_candidate.size-1)

    city_at_a = selected_candidate[point_a]
    city_at_b = selected_candidate[point_b]

    selected_candidate[point_a] = city_at_b
    selected_candidate[point_b] = city_at_a

    return selected_candidate


# a method for crossing over the candidates
def crossover(selected_candidate_a, selected_candidate_b):

    point_a = np.random.randint(1, selected_candidate_a.max())
    point_b = np.random.randint(1, selected_candidate_b.max())

    # get the chromos from a and b
    candidate_a_a = selected_candidate_a[point_a]
    candidate_a_b = selected_candidate_a[point_b]

    # repeat for b
    candidate_b_a = selected_candidate_b[point_a]
    candidate_b_b = selected_candidate_b[point_b]

    return candidate_a, candidate_b


# select the candidates
def select_candidate(fitness_array, pop_size, final):
    # of all candidates, score them using tournament selection
    if final: k = pop_size
    if not final: k = 7
    # loop over and randomly select candidates, return the fittest candidate
    first_candidate = np.random.randint(0, pop_size)
    fittest_candidate = fitness_array[first_candidate]
    candidate_index_to_return = 0
    for i in range(0, k):
        selected_candidate = fitness_array[i]
        if selected_candidate < fittest_candidate:
            print('fitter')
            fittest_candidate = selected_candidate
            candidate_index_to_return = i

    return fittest_candidate, candidate_index_to_return


if __name__ == '__main__':

    # a genetic algorithm to solve complex path finding problems #
    # load dataset
    dataset = pd.read_csv(r'S:\CarbonTrading\IT\git\cd_scripts\Cities\cities.csv')
    dataset['sum_of_distance_x_y'] = np.abs(dataset['X'] - dataset['Y'])
    # convert the dataset to tuple for optimal processing
    dataset_tuple = [tuple(x) for x in dataset.values]

    # hyperparams
    bits = dataset.shape[0]
    pop_size = 100
    generations = 10
    mutation_prob = 0.99
    crossover_prob = 1.01

    # create a random population of size pop size and random cities
    population = create_population(pop_size, bits)

    # fitness_record
    best_candidates_at_gen_x = []
    # perform genetic search
    for i in range(0, generations):
        # used to store our temporary population
        temp_population = np.zeros(shape=(100, bits))
        for j in range(0, pop_size):
            # generate a probability between 0 - 1
            p = random.uniform(0, 1)
            if p <= mutation_prob:
                # select a random candidate
                random_selection = random.randint(0,99)
                selected_candidate = population[random_selection]
                # evolve the candidate
                evolved_candidate = mutation(selected_candidate)
                # assign the new candidate to our temporary population
                temp_population[random_selection] = evolved_candidate

            # if p >= mutation_prob:
            #     # select two random candidates
            #     random_candidates = random.sample(range(100), 2)
            #     selected_candidate_a = population[random_candidates[0]]
            #     selected_candidate_b = population[random_candidates[1]]
            #     # crossover the candidates
            #     candidate_a, candidate_b = crossover(selected_candidate_a, selected_candidate_b)

        # evaluate the candidates, and choose the best
        temp_array = np.array(temp_population)
        fitness_array = calculate_fitness(dataset_tuple, temp_array, pop_size, i)
        fittest_candidate, candidate_index = select_candidate(fitness_array, pop_size, False)
        population[candidate_index] = fittest_candidate
        best_candidates_at_gen_x.append([i, fittest_candidate])

    # split results out to a csv # get the fittest canddiate
    fittest_candidate, candidate_index = select_candidate(fitness_array, pop_size, True)
    candidate = population[candidate_index]
    candidate_df = pd.DataFrame(data=candidate, columns=['Path'])
    candidate_df.to_csv(r'S:\CarbonTrading\IT\git\cd_scripts\Cities\cities_submission.csv', index=False)



