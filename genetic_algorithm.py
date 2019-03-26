import random
import numpy as np
from experiment_methods import *
import copy


def bool2int(x):
    """
    Shift bits and add them to get the total of the gene
    
    Parameters
    ----------
    
    x : numpy array
        This is the array that holds the 1s and 0s that make up the gene.
    """
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y


def create_population(population_size, optimization_constraints, gene_length, introduce_population_size=None):

    if introduce_population_size == None:
        # Creating the population, also known as creating genotype
        population_chromosomes = np.array([random.randint(0,1) for x in range(len(optimization_constraints.keys())*gene_length)])
        for i in range(population_size-1):
            agent_chromosome = np.array([random.randint(0,1) for x in range(len(optimization_constraints.keys())*gene_length)])
            population_chromosomes = np.vstack((population_chromosomes, agent_chromosome))
    else:
        # Creating the population, also known as creating genotype
        population_chromosomes = np.array([random.randint(0,1) for x in range(len(optimization_constraints.keys())*gene_length)])
        for i in range(introduce_population_size-1):
            agent_chromosome = np.array([random.randint(0,1) for x in range(len(optimization_constraints.keys())*gene_length)])
            population_chromosomes = np.vstack((population_chromosomes, agent_chromosome))
    return population_chromosomes

def create_precisions(optimization_constraints, gene_length):
    optimization_precisions = {}
    for key, constraint_range in optimization_constraints.items():
        optimization_precisions[key] = (constraint_range[1]-constraint_range[0])/((2**gene_length)-1)
        
    return optimization_precisions

def create_phenotypes(population_chromosomes, optimization_precisions, gene_length):
    
    #This initial is just the first phenotype array for the first parameter
    phenotypes = np.asarray([bool2int(x) for x in population_chromosomes[:, :gene_length]])
    for i in range(len(optimization_constraints.keys())-1):
        index = (i+1)*gene_length # need to shift the window of parameters we are 
                                  # we are looking at and therefore this index 
                                  # exists
        # We need to decode and stack the phenotypes.
        phenotypes = np.vstack((phenotypes, np.asarray([bool2int(x) for x in population_chromosomes[:, index:(index+gene_length)]])))
    
    #### Decode, also known as phenotype
    index = 0
    for key, min_max_range in optimization_constraints.items():
        phenotypes[index, :] = (phenotypes[index, :] * optimization_precisions[key]) + min_max_range[0]
        index += 1
        
    return phenotypes

def create_warrior_phenotypes(chromosome, optimization_precisions, optimization_constraints, gene_length):
        #This initial is just the first phenotype array for the first parameter
    phenotypes = [bool2int(chromosome[:gene_length])]
    for i in range(len(optimization_constraints.keys())-1):
        index = (i+1)*gene_length # need to shift the window of parameters we are 
                                  # we are looking at and therefore this index 
                                  # exists
        # We need to decode and stack the phenotypes.
        phenotypes.append(bool2int(chromosome[index:(index+gene_length)]))
    
        #### Decode, also known as phenotype
    index = 0
    for key, min_max_range in optimization_constraints.items():
        phenotypes[index] = (phenotypes[index] * optimization_precisions[key]) + min_max_range[0]
        index += 1
        
    return phenotypes


def cross_over(parent1_dict, parent2_dict, probability_of_crossover):
    # probability of crossover is used as a threshold
    num = np.random.rand()

    parent1_chromosome = parent1_dict["genotype"]
    parent2_chromosome = parent2_dict["genotype"]

    if num < probability_of_crossover:

        
        segment_indices = sorted(np.random.choice(len(parent1_chromosome), 2, replace=False))
        
        parent1_segment = copy.deepcopy(parent1_chromosome[segment_indices[0]:segment_indices[1]])
        parent2_segment = copy.deepcopy(parent2_chromosome[segment_indices[0]:segment_indices[1]])
        
        parent1_chromosome[segment_indices[0]:segment_indices[1]] = parent2_segment
        parent2_chromosome[segment_indices[0]:segment_indices[1]] = parent1_segment
        
    return parent1_chromosome, parent2_chromosome
        
def mutation(child1_after_cr, child2_after_cr, probability_of_mutation):
    
    for i in range(len(child1_after_cr)):
        # if a random number is less than the probability of mutation, mutate
        num = np.random.rand()
        if num < probability_of_mutation:
            child1_after_cr[i] = child1_after_cr[i] ^ 1
        
        num = np.random.rand()
        if num < probability_of_mutation:
            child2_after_cr[i] = child2_after_cr[i] ^ 1

    return child1_after_cr, child2_after_cr


def tournament_GA_pytorch(model_class, X_train, y_train, X_test, y_test, \
                          optimization_constraints= {"epochs":(10,100)},\
                          probability_of_crossover=.2,\
                          probability_of_mutation=.2, population_size=20,\
                          generations=30, gene_length=15, num_warriors=3,\
                          input_neurons=2048, output_neurons=5, verbose=False):
    
    if "hidden_neurons" not in list(optimization_constraints.keys()):
        optimization_constraints["hidden_neurons"] = (10, 100)
    
    if "epochs" not in list(optimization_constraints.keys()):
        optimization_constraints["epochs"] = (25, 400)
    
    optimization_precisions = create_precisions(optimization_constraints, gene_length)

    # Need a holder for the new population
    new_population = []
    best_warrior = None
    for gen in range(generations):
        print("Generation: {}".format(gen+1))


        if gen ==0:
            population_chromosomes = create_population(population_size, optimization_constraints, gene_length)
        else:
            # new population is at the end of the generation for loop and is a numpy array after
            # gen > 0
            population_chromosomes = create_population((population_size-new_population.shape[0]), optimization_constraints, gene_length)
            population_chromosomes = np.vstack((new_population, population_chromosomes))
            new_population = []


        for family in  range(population_size//2):
            print("Generation: {}".format(gen+1))
            # Need double the amount of warriors since there will be two parents.
            # Kind of want to make this be more than two parents to see what happens
            warriors = np.random.choice(population_chromosomes.shape[0], num_warriors, replace=False)

            warrior_attrs = {}
            scores = []
            for warrior in warriors:

                phenotypes = create_warrior_phenotypes(population_chromosomes[warrior],
                                        optimization_precisions, optimization_constraints, gene_length)
                
                # The for loop below is just in case I wanted to add more params
                # The ordering matters since the first index of phenotypes is going
                # to be the first key in the dictionary and every phenotype
                # afterwards corresponds to the parameters in the order of the
                # dictionary
                phenotype_dict = {}
                index = 0
                for key in optimization_constraints.keys():
                    phenotype_dict[key] = phenotypes[index]
                    index += 1
                    
                model, y_pred, _ = run_pytorch_pipeline(model_class, X_train,y_train, X_test, y_test,\
                                input_neurons=input_neurons, output_neurons=output_neurons,\
                                hidden_neurons=int(round(phenotype_dict["hidden_neurons"])),\
                                epochs=int(round(phenotype_dict["epochs"])),\
                                training_verbose=verbose)
        
                report = report2dict(classification_report(y_test, y_pred))
                score = float(report[" micro avg"]["recall"])
                scores.append(score)
                print("score: {}".format(score))
                warrior_attrs[warrior] = {}
                warrior_attrs[warrior]["score"] = score
                warrior_attrs[warrior]["phenotypes"] = phenotype_dict
                warrior_attrs[warrior]["genotype"] = population_chromosomes[warrior]
                warrior_attrs[warrior]["model"] = model
                warrior_attrs[warrior]["original_index"] = warrior
                warrior_attrs[warrior]["generation"] = gen
                
            
            indices = np.argsort(-np.asarray(scores))[:2] # 2 means 2 parents
            killed_warriors = np.argsort(-np.asarray(scores))[2:] 
            killed_warriors = [warriors[index] for index in killed_warriors]
            for index in killed_warriors:
                np.delete(population_chromosomes, warrior_attrs[index]["original_index"])

            if best_warrior == None:
                best_warrior = warrior_attrs[warriors[indices[0]]]
            else:
                if best_warrior["score"] < warrior_attrs[warriors[indices[0]]]["score"]:
                    best_warrior = warrior_attrs[warriors[indices[0]]]

            print(best_warrior)
            parent1 = warrior_attrs[warriors[indices[0]]]
            parent2 = warrior_attrs[warriors[indices[1]]]

            child1, child2 = cross_over(parent1, parent2, probability_of_crossover)
            child1, child2 = mutation(child1, child2, probability_of_mutation)

            new_population.append(child1.tolist())
            new_population.append(child2.tolist())

        new_population = np.asarray(new_population)

    
    return best_warrior
            



if __name__ == "__main__":
    

    # Optimization Constraints, key: 'name', value: (min, max)
    optimization_constraints = {"hidden_neurons": (10, 100), "epochs": (25, 400)}
    
    
    
    
        
        
        

        
        
