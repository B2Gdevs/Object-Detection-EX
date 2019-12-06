# Object Detection EX <br>


## What is this repo for?
This repo houses the code to analyze the data produced by detectron.  The code you will find here: 
1. Code that maps the ground truth labels of object detection annotations found in json files to annotations of hdf5 files produced by a modified version of roytseng's detectron.  If you don't know this already, a pytorch implementation of detectron is now an official release and movement from roytsengs is bound to happen soon.  However, modification to get the hdf5 files would be necessary if one were to move.  
    * This is done by using the CVJSON library underneath.  I highly recommend one gets familiar with CVJSON to have an easier time with JSON files made for object detection.
2. Experiment pipelines.
    * This is for the classifiers from **scikit-learn** and models made using **pytorch**.  These two pipelines should be able to handle pretty much all classifiers and models.  They will run the experiment, collect the information and save some reports.  
3. Plotting
    * The plotting script is assumed to only be used with TSNE.  The reason for that is Object Detection is High Dimensional data.  TSNE preserves some level of discrimination in higher dimensions to lower dimensions.  **SO if you plan to visualize data, start with TSNE and plotting.py, then figure something out from there.**
4. Optimization Algorithms
    * Specifically grid search and a genetic algorithm (tournament based selection)
5. Reference scripts
    * Refinement (Not totally complete)
    * SVM building
    * Exhaustive experimentation 


## Data
The data that is used for the examples can be found on the Zvezda server at 

* **Training**
    * "**/mnt/BigData/Datasets/Train/Cocoized/mardct_train_mapped_features/boat_mapped_features.hdf5**"
* **Testing**
    * "**/mnt/BigData/Datasets/Validation/Cocoized/mardct_test_mapped_features/boat_mapped_features.hdf5**"

This data has the class labels inside for **each** annotation.  This is so it is easier to manipulate.
    

## Usage to group classes

```python
    from experiment_methods import *
    from cvjson.cvj import CVJ
    from sklearn.utils import shuffle

    # These features have already been mapped
    # Grabbing the training and test features run an expeirment.
    train_labeled_features_path = "/home/ben/Desktop/mapped_features/train_mardct_coco_fine_ipatch/boat_mapped_features.hdf5" 
    test_labeled_features_path = "/home/ben/Desktop/mapped_features/test_mardct_coco_fine_ipatch/boat_mapped_features.hdf5"

    # This is a CVJ object, this comes from the CVJSON library.  I highly recommend getting 
    # used to the library.  It will save you a lot of development time.
    train_cvj = CVJ("/home/ben/Desktop/completed_train_refinement.json")

    # This name_list variable is how the classes are grouped.  You place the names of classes you want to group in their own list and add it to the variable.
    name_list = [["Lanciafino10mBianca", "Lanciafino10m", "Lanciafino10mMarrone", "Lanciamaggioredi10mBianca"],\
                 ["VaporettoACTV"], ["Mototopo"],\
                  ["Motobarca", "Barchino","Patanella", "Topa", "MotoscafoACTV", "Motopontonerettangolare", "Gondola", "Raccoltarifiuti","Sandoloaremi", "Alilaguna", "Polizia", "Ambulanza"]]
    
    # This is very similar to the train_test_split method by scikit-learn, except better for object detection use cases...
    # and we already have a test set, so it doesn't actually split it, but it does group everything correctly.
    X_train, y_train, X_test, y_test, labels, df_train, df_test = get_grouped_data(name_list, train_labeled_features_path, test_labeled_features_path, train_cvj, verbose=True,\
                                                                sample_threshold=0, list_of_ids_to_ignore=None)

    X_train, y_train = shuffle(X_train, y_train, random_state=0)

    # Congrats, you have just made groupings of different classes
```

## Usage to optimize pytorch models

Ok this is **NOT** Neuro-Evolution or any Neural Architecture Search (NAS).  This is hyper parameter optimization.  The questions that this algorithm attempts to solve are **"What is the best amount of hidden neurons?" and "How long should I train for?"**.  There are other ways to do this, but I took scikit-learns gridsearch approach and an evolutionary technique since evolutionary algorithms are beginning to see more of the limelight again.


**Adding to the code above:**
```python

    # The optimization constraints are the min and max range that you would like the algorithm to search.
    # Currently only two parameters are supported, 'hidden_neurons' and 'epochs'
    # Look closely and 'models.logic_net', that is not an instance of a model.  That is a class definition of a model being passed.
    # The reason for that is now all one has to do is define a model in 'models.py' and then pass it to the GA to optimize.
    # Done.
    # ====================================================================================================================
    optimization_constraints = {"epochs" : (25, 1000), "hidden_neurons": (10,100)}
    best_genetic_params_dict = tournament_GA_pytorch(models.logic_net, X_train, y_train, X_test, y_test,\
                                 optimization_constraints=optimization_constraints, generations=30, population_size=30)
    print(best_genetic_params_dict) # This actually has the model stored in it.
    # ====================================================================================================================


    # I will explain how the GA works further down, now on to grid_search.



    # This is an exhaustive search of the hyperparameter space.  It is only useful when you have a pretty good hunch
    # of where your best hyperparameters will be.  Otherwise use the GA.  This will test every combination of hyper parameters given.
    # ====================================================================================================================
    input_neurons = [2048]
    hidden_neurons = [12,15,25]
    output_neurons = [5] # This can be set to whatever, but don't be surprised if a class that doesn't exist
                         # winds up in your report.
    epochs = [36]
    params, report, _ = grid_search_pytorch(models.logic_net, X_train, y_train, X_test, y_test, input_neurons, hidden_neurons, output_neurons, epochs)
    # ======================================================================================================
    
    # Done with the Gridsearch.
```

## What is and why use a Genetic Algorithm?

Simple, it is a optimization technique to search a space that is defined as NP-complete.  Meaning we know there is a solution, however finding one, or finding the best one is difficult or impossible given that we don't have an infinite amount of time to wait for one.  Evolutionary strategies like GA are ways to **approximate** the best solution.  **Reinforcement Learning** is the star child of the past few years when it comes to **hyperparameter optimization** or **Neural Architecture Search**, however, at least by Uber's results, GA's have been able to find some solutions better than reinforcement learning. 

## How does GA work?

Well, one must create the chromosomes, which can be defined as the blueprint to your parameters, person, animal, and etc.  In a chromosome there are genes, each gene specifies how a parameter is expressed.  Think about why I used the 'parameters' pluraly when talking about a chromosome.  That is because there are many parameters in a chromosome, but there is only 1 parameter in a gene.  **What I am about to say is totally not how genetics work, but is good for an analogy.**  Think about how a chromosome makes a person, a gene would make up the left arm, another gene would make up the right, and so on.  So based on that analogy then we can say that a gene in a chromosome is synonymous with a parameter in a set of parameters or more in tune with our use case, 'hidden_neurons' within a set of 'hidden_nuerons' and 'epochs'.

```python
    import numpy as np
    import random

    # The gene_length is how much of the chromosome makes up 1 gene aka 1 parameter
    gene_length = 15
    # I want two genes per chromosomes, one for 'hidden_neurons' and the other
    # for 'epochs'
    population_chromosomes = np.array([random.randint(0,1) for x in range(2*gene_length)])
    population_chromosomes

    # array([0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1])

    # The above is the chromosome.  The first 15 1s and 0s is the gene representing the 'hidden_neuron' parameter.

```

So how does one search a hyperparameter space with chromosomes?  Well, mutation, crossover, and fitness.  If a model produces good results better than others then we take the chromosome of that model and breed it with it's runner up.  Given a probability of having them swap portions of chromosomes is the probability of crossover.  This is how the chromosomes converge to a minimum or maximum.  Constraining the crossover to specific genes is a worthwile venture.  **Right now my code makes them perform crossover over there entire chromosomes. Meaning that 'hidden_neurons' gene can swap with a portion of 'epochs' gene.**

Ok let's think about something intuitive here.  The **gene length** can be longer which means that there are more numbers that can be generated in between the **optimization_constraints** ranges.  Take a further look above at when GA's were introduced in this README to see what I mean by **optimization_constraints**.

That's the basic concept.  Pretty much everything else is arbitrary to implement.  Like how to get the new generation back in the chromosome pool, how to select the best chromosomes (determined by fitness, our case model performance) and etc

### Caveats to GA
Using a high mutation will generate too much randomness and you will never converge to a solution.  The reason for this is that Evolutionary alogrithms naturally explore spaces in all directions.  **By having a high mutation rate, parameter exploration in all directions is frequent**.  

Having a high crossover means that you will converge very fast, however this only is a problem if you do crossover based on specific genes and don't do crossover over the entire chromosome like my algorithm.  This isn't necessarily a bad thing, but it is the same as throwing away possibly better solutions.  **If you crossover based on gene's then it will be up to your mutation probability to get you out of a local minimum**.

