"""
Author: Benjamin Garrard

This script is used to run the experiments of the neural networks and SVM.
However, this script should be clean from any other methods that require 
definitions.  This is because your experiments will become difficult to
manage.  The experiment_methods.py file is the current script to house
the majority of experiment method definitions.  The models.py is the
definitions of the neural networks that can be used in an experiment.
"""

from experiment_methods import *
from cvjson.cvj import CVJ
from sklearn.utils import shuffle
from genetic_algorithm import *
import pandas as pd


def main():
    train_labeled_features_path = "/home/ben/Desktop/mapped_features/train_mardct_coco_fine_ipatch/boat_mapped_features.hdf5" 
    test_labeled_features_path = "/home/ben/Desktop/mapped_features/test_mardct_coco_fine_ipatch/boat_mapped_features.hdf5"

    train_json_path = "/home/ben/Desktop/completed_train_refinement.json"
    train_image_path = "/home/ben/Desktop/zvezda_data/mardct/Completed_mardct_refinement/completed_train_refinement"

    test_json_path = "/home/ben/Desktop/completed_test_refinement.json"
    test_image_path = "/home/ben/Desktop/zvezda_data/mardct/Completed_mardct_refinement/completed_refinement_test_images"

    train_cvj = CVJ(train_json_path, train_image_path)
    # test_cvj = CVJ(test_json_path, test_image_path)

    # This name_list variable is how the classes are grouped.  You place the names of classes you want in a list and add it to the variable.
    name_list = [["Lanciafino10mBianca", "Lanciafino10m", "Lanciafino10mMarrone", "Lanciamaggioredi10mBianca"],\
                 ["VaporettoACTV"], ["Mototopo"],\
                  ["Motobarca", "Barchino","Patanella", "Topa", "MotoscafoACTV", "Motopontonerettangolare", "Gondola", "Raccoltarifiuti","Sandoloaremi", "Alilaguna", "Polizia", "Ambulanza"]]
    

    X_train, y_train, X_test, y_test, labels, df_train, df_test = get_grouped_data(name_list, train_labeled_features_path, test_labeled_features_path, train_cvj, verbose=True,\
                                                                sample_threshold=0, list_of_ids_to_ignore=None)

    X_train, y_train = shuffle(X_train, y_train, random_state=0)

    get_class_counts(y_train)
    get_class_counts(y_test)


    # This found a hidden neuron parameter of 62 and epochs of 439, recall was at 0.59, highest I have seen
    # It takes a while, so run it and do something else.
    # ====================================================================================================================
    # optimization_constraints = {"epochs" : (25, 1000), "hidden_neurons": (10,100)}
    # best_genetic_params_dict = tournament_GA_pytorch(models.logic_net, X_train, y_train, X_test, y_test,\
    #                             optimization_constraints=optimization_constraints, generations=30, population_size=30)
    # print(best_genetic_params_dict)
    # ====================================================================================================================


    # This basically is like the GA, however it is an exhaustive search.  Much longer than the GA if you try to search
    # a parameter space.
    # ====================================================================================================================
    # input_neurons = [2048]
    # hidden_neurons = [12]
    # output_neurons = [6]
    # epochs = [36]
    # params, report, _ = grid_search_pytorch(models.logic_net, X_train, y_train, X_test, y_test, input_neurons, hidden_neurons, output_neurons, epochs)
    # ====================================================================================================================

    # Gamma was best at 0.01
    pipe, svm_pred = run_scikit_pipeline(SVC(gamma=0.01), X_train, y_train, X_test, y_test, save_report={"file_name": "svm"})

    model, nn_pred, y_scores = run_pytorch_pipeline(models.logic_net, X_train, y_train, X_test, y_test, input_neurons=2048, output_neurons=5, hidden_neurons=62, epochs=439, save_report={"file_name": "one_net"})

    intr_table = intersection_table(df_test, svm_pred, nn_pred)
    cls_table = classification_table(df_test, y_test, svm_pred, nn_pred, y_scores, count_above=2, score_threshold=0.02)

    intr_table.to_csv("intsersection_table.csv")
    cls_table.to_csv("classification_table.csv")

    
    df_test = df_test.reset_index()
    visualize_df = df_test[cls_table.duplicated()]
    
    images= visualize_df['image_name']
    visualize_df = df_test[df_test['image_name'].isin(images)]
    
    visualize(visualize_df, test_image_path)
    
    return None

if __name__ == "__main__":
    main()