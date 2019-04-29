from graphviz import Digraph
import pandas as pd
import random
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import numpy as np
from colorama import Fore, Back, Style
import csv


number_of_people = 20 #input
number_of_generations = 30
elite_size = 5
mutation_rate = 0.01

total_score = []
total_score_average = []

input_file = 'tesztmx11.csv'  #csv file with the adjacency matrix
list_graph = []
with open(input_file) as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        list_graph.append(row)

#print(list_graph)

dict_neighbours = {}
for i in range(len(list_graph)):
    neighbours = []
    for j in range(len(list_graph[i])):
        vertice = list_graph[i][j]
        if vertice == 1:
            neighbours.append(j+1)
    dict_neighbours[i+1] = neighbours

#print(dict_neighbours)
number_vertices = len(dict_neighbours)


dict_indiv_colors = {}
for i in range(number_vertices):
    dict_indiv_colors[i+1] = 0

max_color = 50
def color_nodes(individu):
    score_0 = False
    dict_indiv_colors2 = dict_indiv_colors.copy()
    #print(dict_indiv_colors2)
    for vertice in individu:
        color = 1
        neighbours = dict_neighbours[vertice]

        while color < max_color:
            list_neighbourscolor = []
            for neigh in neighbours:
                list_neighbourscolor.append(dict_indiv_colors2[neigh])
            if color not in list_neighbourscolor:
                dict_indiv_colors2[vertice] = color
                break;
            else:
                color += 1

        if color == max_color:
            score_0 = True

    if score_0:
        return(99)

    else:
        score = max(dict_indiv_colors2.values())
        return(score)


#individual1 = [i+1 for i in range(number_vertices)]

#print(color_nodes(individual1))


def initialisation():
    sample_pop = [i+1 for i in range(number_vertices)]
    sample_pop = random.sample(sample_pop, len(sample_pop))
    return(sample_pop)


def colorize(individu):
    nb_colorless = 0
    current_color = 0
    for index in individu:
        neighbours = dict_neighbours[index]
        non_neighbours = [i+1 for i in range(number_vertices) if i+1 not in neighbours]

    return(nb_colorless)


def firstGeneration():
    dict_score_series = {}
    score_list = []
    pop_list = []
    for i in range(number_of_people):
        series = initialisation()
        score = color_nodes(series)
        score_list.append(score)
        dict_score_series[score] = series
        pop_list.append(series)
    return(score_list, dict_score_series, pop_list)

#score_list, dict_score_series, pop_list = firstGeneration()

#print(dict_score_series)


def elitism(score_list, dict_score_series, pop_list, elitesize):
    new_pop = []
    score_pop_list = []
    total_pop_nb = len(score_list)
    sorted_score_list = sorted(score_list)

    for i in range(elitesize):
        score_pop = sorted_score_list.pop(0)
        elite_pop = dict_score_series[score_pop]
        new_pop.append(elite_pop)
        score_pop_list.append(score_pop)

    for i in range(total_pop_nb - elitesize):
        score_pop = random.choice(sorted_score_list)
        pop_loc = dict_score_series[score_pop]
        new_pop.append(pop_loc)
        score_pop_list.append(score_pop)

    return(new_pop, score_pop_list)


def breed(parent1, parent2): #Uniform crossing
    child1 = []
    child2 = []

    length_gene = len(parent1)
    length_random = int(len(parent1)*random.random())

    child1 = random.sample(parent1[:length_random],len(parent1[:length_random])) + parent1[length_random:]
    child2 = parent2[:length_random] + random.sample(parent2[length_random:],len(parent2[length_random:]))

    return(child1, child2)


def breeding(new_pop,elitesize):
    children = []
    total_pop_nb = len(new_pop)
    random_pop = random.sample(new_pop, len(new_pop))

    for i in range(elitesize):
        children.append(new_pop[i])

    for i in range(int((total_pop_nb - elitesize)/2)):
        parent1,parent2 = random_pop.pop(0),random_pop.pop(0)
        child1, child2 = breed(parent1,parent2)
        children.append(child1)
        children.append(child2)

    return(children)


def mutation(children, mutationrate, elitesize):
    for i in range(elitesize, len(children)):
        pop = children[i]
        for index_gene in range(len(pop)):
            if random.random() < mutationrate:
                random_gene = int(random.random()*len(pop))
                gene_mutate, gene_modified = pop[random_gene], pop[index_gene]
                pop[index_gene], pop[random_gene] = gene_mutate, gene_modified

    return(children)


def nextGeneration(score_list, dict_score_series, pop_list, elitesize, mutationrate):
    #print('PARENTS : ', pop_list)
    #print('SCORE PARENTS : ', score_list)
    new_pop, new_score_pop_list,  = elitism(score_list, dict_score_series, pop_list, elitesize)
    children = breeding(new_pop, elitesize)
    #print('CHILDREN : ', children)
    nextGeneration = mutation(children, mutationrate, elitesize)
    #print('MUTATED CHILDREN : ', nextGeneration)
    new_score_list, new_dict_score_pop = evaluate(nextGeneration)
    return(new_score_list, new_dict_score_pop, nextGeneration)

def evaluate(new_generation):
    new_dict_score_pop = {}
    new_score_list = []
    for pop in new_generation:
        score = color_nodes(pop)
        new_score_list.append(score)
        new_dict_score_pop[score] = pop

    #print("Score list : ", new_score_list)
    min_score = min(new_score_list)
    min_pop = new_dict_score_pop[min_score]
    total_score.append(min_score)
    average_list = sum(new_score_list)/len(new_score_list)
    total_score_average.append(average_list)
    #print('The fastest way is : %s' % min_score)
    #print('Its series is : %s' % min_pop)
    return(new_score_list, new_dict_score_pop)

def fitness():
    score_fitness = 0



def start_simu():

    print("SUMMARY")
    print("Graph file : %s \nSize of the population : %s \nNumber of generations : %s \nElite size : %s \nMutation rate : %s" % (input_file, number_of_people, number_of_generations, elite_size, mutation_rate))

    print("\nRunning the program...")
    start_time = time.time()

    score_list, dict_score_series, pop_list = firstGeneration()

    end_time1 = time.time()
    timer = end_time1 - start_time
    approx_time = timer * number_of_generations
    print("It will take approximately %s seconds (%s minutes)" % (str(round(approx_time, 2)), str(round(approx_time/60, 1))))

    #print("First population score : ", sorted(score_list))
    average_list = sum(score_list)/len(score_list)
    print("Average Score of the first population : ", average_list)

    pbar = tqdm([i for i in range(number_of_generations)])


    for i in range(number_of_generations):
        score_list, dict_score_series_plus, pop_list = nextGeneration(score_list, dict_score_series, pop_list, elite_size, mutation_rate)
        dict_score_series.update(dict_score_series_plus)
        pbar.update()
        pbar.set_description("Processing generation %s" % str(i+1))

    print("Program done")

    end_time2 = time.time()
    timer = end_time2 - start_time
    print("It has taken %s seconds" % str(round(timer, 2)))

    #print("Last population score : ", sorted(score_list))
    average_list = sum(score_list)/len(score_list)
    print("Average Score of the last population: ", average_list)

    #print("SCORE AVERAGE : ", total_score_average)
    #print("TOTAL SCORE : ", total_score)
    try :
        best_score = min(total_score)
        print("Fastest found : " + '\033[92m' + str(best_score))
        print(Style.RESET_ALL)
        print("Best pop is : ", dict_score_series[best_score])
    except:
        print("Error with the dictionary")
        pass;


    plt.plot(total_score, 'r', label='Best score of a generation')
    #plt.plot(total_score_average, 'g', label='Average score of a generation')
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.legend()
    plt.savefig('graph_score.png')
    plt.show()


start_simu()