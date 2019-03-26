from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
import h5py
import numpy as np
import matplotlib as plt
from cvjson.cvj import CVJ
import os
import copy
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import plotting
import torch
import torch.nn as nn
import torchvision
from models import hdf_Dataset
import models
import cv2

def get_data(hdf5_path, list_of_classes_to_keep=None, semantics=False, negatives=False, list_of_ids_to_ignore=None):
    """
    This method takes a path to a single hdf file and will loop through it and build the data for the file.

    Parameters
    ----------
    hdf5_path : string
            The path to a single hdf file to be split in to the many classes it has stored in it.

    list_of_classes: list
            This is a list of classes found by the run_classifer method.  These are K top classes that
            have the most annotations.

    semantics: bool
            This variable is was says to return the semantic vectors produced by detectron is a list_of_lists of the deep features produced by detectron.  
            A semantic vector is also known as the softmax vector produced by detectron in this case.  It is just the probabilities of each class
            in vector from.  This is useful since one can look at the vectors enough time to point out a pattern of probabilities that actually belong
            to a specific class.


    negatives: bool
            This variable, if set to True, will turn every annotation this method finds to a negative class with the 
            ID=5000, and the class="negative"


    Returns
    -------
    X: numpy array
            Array of X values

    Y: numpy array
            Array of Y values

    category_names: numpy array
            Array of category names associated with the class IDs
            
    """
    image_name_y_value = []
    X = []
    Y = []
    file_ = h5py.File(hdf5_path)

    for image, values in list(file_["frames"].items()):

        boxes = values["bounding_boxes"]
        features = values["descriptors"]
        features = np.asarray(features)
        scores = values["scores"]
        scores = np.asarray(scores)

        try: 
            features = features.reshape(scores.shape[0], -1)
            scores = scores.reshape(scores.shape[0], scores.shape[1])
        except ValueError:
            print("A value error occurred in get_data() when reshaping the features.  Continuing.")
            

        if negatives == False:
            categories = values["categories"]
            categories = np.asarray(categories).astype(int)
        else:
            categories = np.asarray([5000 for i in range(features.shape[0])]) # 5000 is the id I am using for the negative class
            list_of_classes_to_keep = [5000]

        count_ = 0
        for cat in categories:
            image_name_y_value.append((image, cat, boxes[count_]))
            count_ += 1
            

        ##### START#### This iterates through and makes the X and Y data
        if len(X) == 0:
            
            if list_of_classes_to_keep != None and semantics == True : # if semantics is true then you will get the semantic vector.  AKA the scores that were predicted
                for index, cat in enumerate(categories):
                    if cat in list_of_classes_to_keep:
                        X.append(np.copy(scores[index]))
                        Y.append(np.copy(categories[index]))
                        

            elif list_of_classes_to_keep == None and semantics == True:
                X = np.copy(scores)
                Y = np.copy(categories)

            elif list_of_classes_to_keep != None and semantics == False: 
                for index, cat in enumerate(categories):
                    if cat in list_of_classes_to_keep:
                        X.append(np.copy(features[index]))
                        Y.append(np.copy(categories[index]))
            else :
                X = np.copy(features)
                Y = np.copy(categories)

            if len(X) != 0:
                X = np.asarray(X)
                Y = np.asarray(Y)



        else:
            if list_of_classes_to_keep != None and semantics == True: # if semantics is true then you will get the semantic vector.  AKA the scores that were predicted
                for index, cat in enumerate(categories):
                    if cat in list_of_classes_to_keep:
                        try:
                            Y = np.vstack((Y, cat))
                            X = np.vstack((X, scores[index]))
                        except ValueError:
                            try:
                                Y.shape[1]
                            except IndexError:
                                Y = Y[-1]
                                X = X[-1]

            elif list_of_classes_to_keep == None and semantics == True:
                for index, cat in enumerate(categories):
                        try:
                            Y = np.vstack((Y, cat))
                            X = np.vstack((X, scores[index]))
                        except ValueError:
                            try:
                                Y.shape[1]
                            except IndexError:
                                Y = Y[-1]
                                X = X[-1]

            elif list_of_classes_to_keep != None and semantics == False:
                for index, cat in enumerate(categories):
                    if cat in list_of_classes_to_keep:
                        try:
                            Y = np.vstack((Y, cat))
                            X = np.vstack((X, features[index]))
                        except ValueError:
                            try:
                                Y.shape[1]
                            except IndexError:
                                Y = Y[-1]
                                X = X[-1]
            else:
                for index, cat in enumerate(categories):
                        try:
                            Y = np.vstack((Y, cat))
                            X = np.vstack((X, features[index]))
                        except ValueError:
                            try:
                                Y.shape[1]
                            except IndexError:
                                Y = Y[-1]
                                X = X[-1]

        #### END ####
    # print(Y.shape)
    # print(X.shape)
    if list_of_ids_to_ignore != None:
        for id in list_of_ids_to_ignore:
            ind = np.where(Y[:] != id)[0] 
            Y = Y[ind, :]
            X = X[ind, :] 
    counts = np.unique(Y, return_counts=True)
    # print(counts[0])
    # print(Y.shape)
    # print(X.shape)
    # input()
    
    Y = Y.ravel() # Flattening to a 1-d array
    category_names = np.asarray(file_["class_names"])
    file_.close()

    df = pd.DataFrame(image_name_y_value)
    df.columns = ["image_name", "y_values", 'bbox']
    df = df.set_index("image_name")
    return X, Y, category_names, counts, df
            


#Deprecated don't use.  This was for some experiments that was ill conceived.
def run_classifier(classifier, train_path=None, test_path=None, training_negatives=None, test_negatives=None, train_cvj=None,\
                  test_cvj=None, n_best_class=5, test_path_is_negative=False, semantics=False, norm=None):
    """
    This method runs the classifier with a number of different settings.  This ranges from having files fed in as only being negatives,
    only negatives in the training set, only negatives in the testing set, and no negatives (which since you will most likely be using a multclass svm),
    is what you want.
    """

    if train_cvj != None and n_best_class != 0:
        list_of_class_to_keep = get_top_count_ids(train_cvj, n_best_class)
        print(list_of_class_to_keep)
        input()

    train_positives_count = 0
    train_negatives_count = 0
    if train_path != None:
        X_train, Y_train, categories, train_counts = get_data(train_path, list_of_classes_to_keep=list_of_class_to_keep, semantics=semantics)
        train_positives_count = X_train.shape[0] 
        if training_negatives != None:
            for file_path in training_negatives:
                try:
                    X_neg, Y_neg, _, _ = get_data(file_path, list_of_classes_to_keep=list_of_class_to_keep, negatives=True, semantics=semantics)
                    X_train = np.concatenate((X_train, X_neg))
                    Y_train = np.concatenate((Y_train, Y_neg))
                except ValueError:
                    continue
            train_negatives_count = X_train.shape[0] - train_positives_count
        # print(X_train)
        if norm:
            X_train = normalize(X_train, norm=norm)
        classifier.fit(X_train, Y_train)
        print("\n\nTraining Positive Count = {}, Training Negative Count {}".format(train_positives_count, train_negatives_count)) # honestly don't listen to the negatives count.
        print("\nTraining class counts\n")
        for index, count in np.ndenumerate(train_counts[1]):
            if id != 5000:
                print("class = {}, count = {}".format(train_cvj.get_class_id_2_name(train_counts[0][index]), count))
            else:
                print("class = {}, count = {}".format("negative", count))

    test_positives_count = 0
    test_negatives_count = 0
    if test_path != None and test_path_is_negative != True:
        X_test, Y_test, _, test_counts = get_data(test_path, list_of_classes_to_keep=list_of_class_to_keep, semantics=semantics)
        test_positives_count = X_test.shape[0]

        if test_negatives != None:
            for file_path in test_negatives:
                try:
                    X_neg, Y_neg, _, test_counts = get_data(file_path, list_of_classes_to_keep=list_of_class_to_keep, negatives=True, semantics=semantics)
                    X_test = np.concatenate((X_test, X_neg))
                    Y_test = np.concatenate((Y_test, Y_neg))
                except ValueError:
                    continue
            test_negatives_count = test_positives_count - X_test.shape[0]

    
    elif test_path_is_negative:
        X_test, Y_test, _, test_counts = get_data(test_path, list_of_classes_to_keep=list_of_class_to_keep, negatives=True, semantics=semantics)
        test_positives_count = 0
        test_negatives_count = X_test.shape[0]

    else:
        print("No test files given, returning classifier.")
        return classifier

    print("\nTest class counts\n")
    for index, count in np.ndenumerate(test_counts[1]):
        if id != 5000:
            print("class = {}, count = {}".format(train_cvj.get_class_id_2_name(test_counts[0][index]), count))
        else:
            print("class = {}, count = {}".format("negative", count))

    if norm: # FYI normalizing the deep features will produce terrible results. ALSO normalizing args are "l2", "l1", and "max"
        X_test = normalize(X_test, norm=norm)
    preds = classifier.predict(X_test)
    print("Test Positive Count = {}, Test Negative Count {}".format(test_positives_count, test_negatives_count))
        

    if train_cvj != None:
        list_of_names = [train_cvj.get_class_id_2_name(class_id) for class_id in list_of_class_to_keep]
    else:
        list_of_class_to_keep = [id for id in range(len(categories) + 1) if id != 5] # this is MARDCT specific, remove this is using a different dataset
        list_of_names = categories

    report = classification_report( Y_test, preds, target_names=list_of_names, labels=list_of_class_to_keep)
    print(report)
    print("\n\naccuracy = {}".format(accuracy_score(Y_test, preds, normalize=True)))

    if test_negatives or training_negatives or test_path_is_negative:
        list_of_names.append("negative")
        list_of_class_to_keep.append(5000) # ID I use for negatives

    cm = confusion_matrix(Y_test, preds, labels=list_of_class_to_keep) # create a confustion matrix to be visualized by cm_heatmap

    print_cm(cm, list_of_names) # a Pretty print for the confusion matrix

    return classifier, np.asarray(preds), cm, report, list_of_names



def get_top_count_ids(cvj_obj, top_count=5):
    """
    This method will look in the CVJ object and return the classes with the most
    annotations

    Honestly, this method could be integrated in to the CVJSON library
    
    Parameters
    ----------
    cvj_obj : CVJ object 
            The CVJ object comes from the CVJSON library, this object must hold you

    top_count: int
            This variable contains the number of classes you would like returned.

    Returns
    -------
    list_of_classes : list
            The list of classes with the most annotations in the CVJ object


    """

    class_2_anns = copy.deepcopy(cvj_obj.get_class_id_2_anns())

    list_of_classes = []
    for i in range(top_count):
        key = max(class_2_anns, key= lambda x: len(class_2_anns[x]))

        list_of_classes.append(key)
        del class_2_anns[key]

    return list_of_classes

def print_cm(cm, labels):
    """  
    This method prints the confusion matrix in a way that is readable.
    
    Parameters
    ----------
    cm : numpy array
            This confusion matrix is assumed to be from scikit-learn.

    labels: list
            This is a list of strings that are the labels for the confusion matrix.  These
            will replace the rows and columns with the names of the labels

    Returns
    -------
    cm : numpy array
            This confusion matrix is assumed to be from scikit-learn.
    
    """
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "
    
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES
    
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
        
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.d".format(columnwidth) % cm[i, j]
            print(cell, end=" ")
        print()

    return cm

def cm_heatmap(cm, labels, file_name, show=False):
    """  
    This method shows/saves a heatmap of the confusion matrix with
    seaborn.
    
    Parameters
    ----------
    cm : numpy array
            This confusion matrix is assumed to be from scikit-learn.

    labels: list
            This is a list of strings that are the labels for the confusion matrix.  These
            will replace the rows and columns with the names of the labels

    file_name: string
            This is the file_name or file_path where the plot will be saved.
    
    show: bool
            If this is True then the plot will be shown to the user of this method

    Returns
    -------
    plt : plot object
            This is the plot of the heatmap generated by seaborn.  This is a matplotlib
            plt object
    
    """
    df_cm = pd.DataFrame(cm, index = [i for i in labels],
                  columns = [i for i in labels])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt="d")

    if show:
        plt.show()
        input()
    plt.savefig(file_name)

    return plt

def report2dict(cr):
    """
    This method converts the classification report from scikit-learn 
    to a dictionary.
    
    Parameters
    ----------
    D_class_data : dictionary
            This is the classification report as a dictionary.
    """
    # Parse rows
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)
    
    # Store in dictionary
    measures = tmp[0]

    D_class_data = defaultdict(dict)
    for row in tmp[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
    return D_class_data

def group_categories(X_data, y_data, list_of_lists_of_ids, df, noise=False, noise_id=24):
    """
    This method receives a list of lists of category ids and the
    data from the get_data() method to group them.  Each list of ids
    will be a group.  The features from the X 
    data will be associated with the new labels automatically because the 
    features are associated with the labels based on position.

    x[0] is associated with y[0].  So if y[0] changes it's label then 
    x[0] is now assocatied with that label.

    However, X_data needs to be sent so it can be reduced the dimensions that
    the y_data will be.

    This is done, by refactoring the ids to associate with those ids.

    e.g. The first list of ids will be group 25, their class id will then be 25.

    This is incremented.  Group 25, group 26, group K for K lists of ids found
    in list_of_list_of_ids.  So the ORDER will matter if you want a particular
    class to be in a particular group.

    As the user of this method, it is your responsibility to match these ids to
    their appropriate names outside of this method.  This method ONLY reclassifies
    the data.


    Parameters
    ----------
    X_data : numpy array
            This is the sample data normally known as X_train or X_test

    y_data: numpy array
            This is the label data normally known as y_train or y_test

    list_of_lists_of_ids: list
            This is a list of lists.  This is used for grouping specific id's together to form 
            a singular class out of many classes in the dataset
    
    noise: bool
            If this is True then this will use the noise id specifed as the class that
            is used for noise.  This will make the noise class have the highest ID

    noise_id: int
            see the definition of noise above

    Returns
    -------
    X_data : numpy array
            This is the sample data that has been refactored

    y_data : numpy array
            This is the label data that has been refactored
    """

    group_id = max(max([max(list_) for list_ in list_of_lists_of_ids]), 25) # 24 is the highest id of the MARDCT dataset so 25 should be the max.  However, I wanted to attempt at
                                                                            # making the code reusable.
    start = group_id
    for list_of_ids in list_of_lists_of_ids:
        mask = np.isin(y_data, list_of_ids) # generates a mask of indices, this should work on any dimension
        y_data[mask] = group_id # this sets the data found by the mask to the group number, aka the new class id
        # print(df[df["y_values"].isin(list_of_ids)]["y_values"]) #= group_id
        df.loc[df["y_values"].isin(list_of_ids), "y_values"] = group_id

        group_id += 1

    if noise:
        df.loc[df["y_values"] == noise_id, "y_values"] = group_id
        df = df.loc[df["y_values"] >= start, :]
        y_data[y_data == noise_id] = group_id # group_id will be up by one and no other class should have it
        X_data = X_data[y_data >= start]
        y_data = y_data[y_data >= start]

    else:
        X_data = X_data[y_data >= start]
        y_data = y_data[y_data >= start]
        df = df.loc[df["y_values"] >= start, :]

    # Ridiculuous, if you have any ids that skip increments then you will run in to "src/THCUNN/ClassNLLCriterion.cu:105:"
    # type errors.  This happens because you need the minimum amount of output neurons as there are classes
    # if you have a class id of "25" and just one other, you will still need 26 neurons to satisfy the torch model.
    # so refactoring is necessary.
    unique_ids = sorted(np.unique(y_data).tolist())
    count = 0
    for i in unique_ids:
        y_data[y_data == i] = count
        df.loc[df["y_values"] == i, "y_values"] =  count
        count += 1

    # print(np.unique(y_data))

    return X_data, y_data, df

def keras_logic(X_train, y_train, X_test, y_test, input_shape_tup, class_num):

    """
    so far it is broken
    """
    # Import `Sequential` from `keras.models`
    from keras.models import Sequential

    # Import `Dense` from `keras.layers`
    from keras.layers import Dense, Flatten
    from keras import metrics
    from keras import losses
    from keras.optimizers import SGD, RMSprop

    print(X_train[X_train == np.nan])
    print(X_train[X_train == np.inf])
    # Initialize the constructor
    model = Sequential()

    print(input_shape_tup[1:])
    # Add an input layer 
    model.add(Dense(class_num, activation='softmax', input_shape=input_shape_tup[1:]))

    # Add an output layer 
    # model.add(Dense(class_num, activation='softmax'))

    model.compile(optimizer=RMSprop(lr=.000001), loss=losses.sparse_categorical_crossentropy, metrics=[metrics.categorical_accuracy])
    print(model.summary())
    model.fit(X_train, y_train, batch_size=1, epochs=2, shuffle=False, verbose=2)
    return model


def train_torch(model_class, X_train, y_train, batch_size=30, input_neurons=None, hidden_neurons=500, \
                output_neurons=None, learning_rate=.01, epochs=400, verbose=True):
    """

    This method is the driving method for training a pytorch model.

    Parameters
    ----------
    X_train : numpy array
            This is the sample data

    y_train: numpy array
            This is the label data 

    batch_size: int
            This is the size of how many samples each iteration will use
    
    input_neurons: int
            This is the size the of the features
    
    hidden_neurons: int
            This is the size the of layers in between the input and output layers

    output_neurons: int
            This is the size of the output neurons

    learning_rate: float
            This is the learning rate of the neural network. This handles how fast 
            the network will converge to a solution

    epochs: int
            This is the amount of times the network will iterate through the ENTIRE 
            data set.


    Returns
    -------
    model : nn.Module
            this is the model of the trained network

    """
    
    train = hdf_Dataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)

    # interestingly if the output neurons are at 10 even though there
    # are only 5 classes, errors occur in the loss.  Cuda assert NLLCriterion error.
    if input_neurons == None:
        input_neurons = X_train.shape[1]

    if output_neurons == None:
        output_neurons = len(np.unique(y_train).tolist())

    model = model_class(input_neurons, hidden_neurons, output_neurons) 
    # print(model.fc1.weight)# to make sure the weights are always the same for real reproducibility and debugging.
    # print(model.fc2.weight)
    # print(model.fc3.weight)
    # input()

    gpu_available = torch.cuda.is_available()
    if gpu_available:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    correct_during_training = 0
    total_during_training = 0
    for i in range(epochs):
        for k, (arrays, labels) in enumerate(train_loader):

            arrays = torch.autograd.Variable(arrays)
            labels = torch.autograd.Variable(labels)

            if gpu_available:
                arrays = arrays.cuda()
                labels = labels.cuda()

            outputs = model(arrays)
            pred = torch.max(outputs, 1)[1]


            if gpu_available:
                correct_during_training += (pred.cpu() == labels.cpu()).sum()
                total_during_training += labels.size(0)
            else:
                correct_during_training += (pred == labels).sum()
                total_during_training += labels.size(0)


            loss = criterion(outputs, labels)
            loss.backward()

    
            optimizer.step()
            optimizer.zero_grad()

            percentage = (100 * (correct_during_training.float()/total_during_training))


            if verbose:
                if k % 100 == 0:
                    print("loss = {},  Correct = {}, Total = {},  Percentage = {}".format(loss, correct_during_training, total_during_training, percentage))
        
    return model


def test_torch(model, X_test, y_test, batch_size=30):
    """
    This method is the driving method for testing a pytorch model.

    Parameters
    ----------
    model: nn.Module
            This is the model that is created when using the nn.Module class from torch.

    X_test : numpy array
            This is the sample data

    y_test: numpy array
            This is the label data 

    batch_size: int
            This is the size of how many samples each iteration will use

    Returns
    -------
    model : nn.Module
            This is the model of the trained network
    
    y_pred: numpy array
            This is the predictions of the network in the order of the X_test set.

    """

    test = hdf_Dataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size)

    gpu_available = torch.cuda.is_available()
    if gpu_available:
        model = model.cuda()

    for i in range(1):
        y_pred = []
        for k, (arrays, labels) in enumerate(test_loader):

            arrays = torch.autograd.Variable(arrays)
            labels = torch.autograd.Variable(labels)

            if gpu_available:
                arrays = arrays.cuda()
                labels = labels.cuda()

            outputs = model(arrays)
            pred = torch.max(outputs, 1)[1]

            if gpu_available:

                if len(y_pred) == 0:
                    y_pred = pred.cpu().numpy()
                    y_scores = outputs.data.cpu().numpy()
                    
                    
                else:
                    y_pred = np.hstack((y_pred, pred.cpu().numpy()))
                    y_scores = np.vstack((y_scores, outputs.data.cpu().numpy())) 

    return model, y_pred, y_scores

def generate_noise(hdf_file_path, semantics=False):
    """
    This method needs to be given the hdf file that you wish to be noise.
    Regardless of what classes are in the file this will label them all as 5000
    which is an arbitrarily defined noise class

    Parameters
    ----------
    hdf_file_path: string
            This is the file path to the hdf5 file that should be considered noise

    semantics : bool
            If true this will use the scores in the hdf5 file as the features.
    """
   
    file_ = h5py.File(hdf_file_path)

    noise = []
    for image, values in list(file_["frames"].items()):

        # boxes = values["bounding_boxes"]
        features = values["descriptors"]
        features = np.asarray(features)
        scores = values["scores"]
        scores = np.asarray(scores)

        try: 
            features = features.reshape(scores.shape[0], -1)
            scores = scores.reshape(scores.shape[0], scores.shape[1])
        except ValueError:
            print(features.shape)
            print(features)

        if len(noise) == 0:
            if not semantics:
                noise = np.copy(features)
            else:
                noise = np.copy(scores)
    
            if len(noise) != 0:
                noise = np.asarray(noise)

        else:
            if not semantics:
                for index in range(features.shape[0]):
                    noise = np.vstack((noise, features[index]))
            else:
                for index in range(features.shape[0]):
                    noise = np.vstack((noise, scores[index]))

    labels = np.zeros((noise.shape[0],))
    labels[:] = 5000 # using 5000 as my negatives/noise class.  The noise data should be any detections that did not correspond to
                     # a ground truth label.  This is the noise that is produced by the model.

    labels = labels.ravel()
    return noise, labels, "noise" # noise is the label for this data.

def get_grouped_data(name_list, train_labeled_features_path, test_labeled_features_path, train_cvj,\
                     verbose=True, save_plot=None, list_of_ids_to_ignore=None, sample_threshold=50, noise_id=24): 
    """
    This method needs to be given the hdf file that you wish to be noise.
    Regardless of what classes are in the file this will label them all as 5000
    which is an arbitrarily defined noise class

    Parameters
    ----------
    name_list: list
            This is a list of lists.  These are the names of the classes to be grouped together.

    train_labeled_features_path : str
            This is the file path to the hdf5 file created when using the map_labels.py file.  
            Speficially this is the file path to the training set of the file created with the labels.py file

    test_labeled_features_path: str
            This is the file path to the hdf5 file created when using the map_labels.py file.  
            Speficially this is the file path to the test set of the file created with the labels.py file

    train_cvj: CVJ object
            This is CVJ object created using the training set.  The CVJ object requires the JSON file
            of the training set.  

    verbose: bool
            If this is true then this method will ouput to the console what it removed, what the ids were,
            what is left and what the ids are.  Sometimes not every class makes it in to the list due to the
            sample threshold

    save_plot: dict
            If the plots are wanted then this will use TSNE to map the dimensions to a lower dimension.  
            The keys that must be set are "dimension" and "file_name" which indicates which dimension to map to and
            where to save the plot respectively.

    list_of_ids_to_ignore: list
            The ids of the data inside this list will be used to completely ignore them during training and testing.

    sample_threshold: int
            The sample threshold works by removing any classes that do not have a sample size of K found in the training
            set.  This means that if there are samples_amount[class] > K, the test set could be less than K, but the 
            class still appear in the reports

    noise_id: int
        This is used to identify the noise class if there is one.  In Mardct the noise class is 24 for "other" this class
        is also referred to as the confusion class.

    Returns
    -------
    X_train : numpy array
            This is the refactored sample data from the training set.

    y_train : numpy array
            This is the refactored label data from the training set.

    X_test : numpy array
            This is the refactored sample data from the test set.

    y_test : numpy array
            This is the refactored label data from the test set.

    labels: numpy array
            These are the unique labels found in the refactored training set.
    """
    id_list = []
    removed_list = []
    for names in name_list:
        group = []
        for name in names:
            idx = train_cvj.get_class_name_2_id()[name.lower()]
            if len(train_cvj.get_class_id_2_anns(idx)) > sample_threshold:
                group.append(idx)
            else:
                removed_list.append(name)
                    
        id_list.append(group)
    id_list = [list_ for list_ in id_list if len(list_) > 0]

    new_list= []
    for i in id_list:
        new_list.extend(i)
    class_list = [train_cvj.get_class_id_2_name(i) for i in new_list]

    if verbose:

        print("The ID's of the groups are as follows:")
        print(id_list)
        print("\nThe following classes were removed due to having"\
         " a TRAINING sample size less than {}".format(sample_threshold))
        print(removed_list)
        print("\nThe classes left are as follows:")
        print(class_list)
        print(new_list)
    dif = set(train_cvj.get_category_ids()) - set(new_list)
    if noise_id != None:
        dif.remove(noise_id)


    X_train, y_train, cats, counts, df_train = get_data(train_labeled_features_path, list_of_ids_to_ignore=list_of_ids_to_ignore)
    X_test, y_test, cats, count_, df_test = get_data(test_labeled_features_path, list_of_ids_to_ignore=list_of_ids_to_ignore)
    
    classes, counts = counts
    dif = list(dif)

    for index, class_ in np.ndenumerate(classes):
        if counts[index] < sample_threshold:
            if class_ not in dif:
                dif.append(class_)

    df_train = df_train[~df_train["y_values"].isin(dif)]
    df_test = df_test[~df_test["y_values"].isin(dif)]
    X_train = X_train[np.isin(y_train, dif, invert=True)]
    y_train = y_train[np.isin(y_train, dif, invert=True)]
    
    X_test = X_test[np.isin(y_test, dif, invert=True)]
    y_test = y_test[np.isin(y_test, dif, invert=True)]
    

    if save_plot != None:
        try:
            dimension = save_plot["dimension"]
            if dimension == 3:
                threeD_model = TSNE(n_components=3, random_state=0)
                threeD_tsne_matrix = threeD_model.fit_transform(X_train)
                threeD_df = plotting.threeD_data(threeD_tsne_matrix, y_train)
                plotting.plot_3d(threeD_df, file_name=save_plot["file_name"], jupyter=False)
            elif dimension == 2:
                twoD_model = TSNE(n_components=2, random_state=0)
                twoD_tsne_matrix = twoD_model.fit_transform(X_train)
                twoD_df = plotting.twoD_data(twoD_tsne_matrix, y_train)
                plotting.plot_3d(twoD_df, file_name=save_plot["file_name"], jupyter=False)
        except KeyError:
            print("save_plot variable needs to have keys \'dimension\' and \'file_name\'.")

    X_train, y_train, df_train = group_categories(X_train, y_train, id_list, df_train, noise=True)
    X_test, y_test, df_test = group_categories(X_test, y_test, id_list, df_test, noise=True)
    labels = np.unique(y_train)

    return X_train, y_train, X_test, y_test, labels, df_train, df_test

def run_scikit_pipeline(classifier, X_train, y_train, X_test, y_test,\
                             labels=[], save_report={}, console_report=True, grid_search_params=None):

    """
    This method needs to be drives the classification experiments using scikit-learn.  It is a modular
    piece of code in that it should be able to take any classifier from scikit-learn as long as it fits
    the correct parameters.

    Parameters
    ----------
    classifier: estimator
            This is a list of lists.  These are the names of the classes to be grouped together.

    X_train: numpy array
            This is the sample data of the training set

    y_train : numpy array
            This is the label data of the training set

    X_test: numpy array
            This is the sample data of the test set

    y_test: numpy array
            This is label data of the test set 

    labels: list
            This is the list of unique labels that are in the dataset

    save_report: dict
            If the report is to be saved this dictionary must contain the key "file_name"
            along with the file_name/file_path as the value

    console_report: bool
            If this is True then a report will be generated to the console.

    grid_search_params: dict
            This is used for searching for the best possible parameters using gridsearch.  Right now the classifier's name will be
            clf due to standard naming conventions.  So if you wanted to search the gamma values of an SVM in the dictionary you would put
            {'clf__gamma': [.1, .2, .3]}.  Continue adding any parameters you wish to search.  Just remember the double underscore.
            This also is used to set parameters for something other than default. As long as the variable to have a custom parameter
            isn't given more than one, that parameter passed is the one that will be used.  Therefore making this pipeline modular
            and able to fit most classifiers by scikit-learn.

    Returns
    -------
    pipe: estimator
            This is technically a pipe object, but is also the classifier

    preds: numpy array
            These are the predictions made by the classifier
    """
    # Sci-kit pipelines take a list of named estimators (just means dictionaries), in pytorch this would be known as transforms, and then performs the estimator
    # functions.  An estimator can be PCA or it could be a model.

    estimators = [('clf', classifier)]
    from sklearn.model_selection import GridSearchCV

    pipe = Pipeline(estimators)

    if grid_search_params != None:
        pipe = GridSearchCV(pipe, grid_search_params)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    print(y_test)
    print(preds)

    if len(labels) == 0:
        labels = np.unique(y_test).tolist()

    if console_report or save_report:
        cls_report = classification_report(y_test, preds)
        cm = confusion_matrix(y_test, preds, labels=labels) # create a confustion matrix to be visualized by cm_heatmap

    if console_report:
        if grid_search_params != None:
            print("best params")
            print(pipe.best_params_)

        labels = [str(x) for x in labels]
        print("Confusion Matrix")
        print_cm(cm,labels)
        print("Classification report")
        print(cls_report)

    if save_report != {}:
        file_name = save_report["file_name"]
        cm_heatmap(cm, labels, file_name)
        csv_file_name = "cls_report_" + save_report["file_name"] + ".csv" 

        df = pd.DataFrame.from_dict(report2dict(cls_report))
        print(df)
        df.to_csv(csv_file_name)
    
    return pipe, preds

def run_pytorch_pipeline(model_class, X_train, y_train, X_test, y_test,\
                             labels=[], save_report={}, \
                             console_report=True, input_neurons=2048,\
                             hidden_neurons=100, output_neurons=250, epochs=1, training_verbose=True):
    """
    This method is the driving method for training/testing a model and saving the report.
    This however does not save the model.

    Parameters
    ----------
    X_train: numpy array
            This is the sample data of the training set

    y_train : numpy array
            This is the label data of the training set

    X_test: numpy array
            This is the sample data of the test set

    y_test: numpy array
            This is label data of the test set

    labels: list
            This is the list of unique labels that are in the dataset

    save_report: dict
            If the report is to be saved this dictionary must contain the key "file_name"
            along with the file_name/file_path as the value

    console_report: bool
            If this is True then a report will be generated to the console.

    input_neurons: int
            This is the size the of the features
    
    hidden_neurons: int
            This is the size the of layers in between the input and output layers

    output_neurons: int
            This is the size of the output neurons

    epochs: int
            This is the amount of times the model will iterate over the ENTIRE dataset.

    Returns
    -------
    model : nn.Module
            This is the model of the trained network
    
    y_pred: numpy array
            These are the predictions made.

    """
    model = train_torch(model_class, X_train, y_train, input_neurons=input_neurons,\
                        hidden_neurons=hidden_neurons, output_neurons=output_neurons,\
                        epochs=epochs, verbose=training_verbose)
    model, y_pred, y_scores = test_torch(model, X_test, y_test)

    if len(labels) == 0:
        labels = np.unique(y_test)

    if console_report or save_report:
        cls_report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=labels) # create a confustion matrix to be visualized by cm_heatmap


    if console_report:
        print("Confusion Matrix")
        labels = [str(x) for x in labels]
        print_cm(cm,labels)
        print("Classification report")
        print(cls_report)

    if save_report != {}:
        file_name = save_report["file_name"]
        cm_heatmap(cm, labels, file_name)
        csv_file_name = "cls_report_" + save_report["file_name"] + ".csv" 

        df = pd.DataFrame.from_dict(report2dict(cls_report))
        print(df)
        df.to_csv(csv_file_name)

    return model, y_pred, y_scores


def grid_search_pytorch(model_class, X_train, y_train, X_test, y_test,\
                        input_neurons, hidden_neurons, output_neurons,\
                        epochs, verbose=False):


    """
        This method is the driving method for training/testing a model and saving the report.
    This however does not save the model.

    Parameters
    ----------
    X_train: numpy array
            This is the sample data of the training set

    y_train : numpy array
            This is the label data of the training set

    X_test: numpy array
            This is the sample data of the test set

    y_test: numpy array
            This is label data of the test set

    input_neurons: int
            This is the size the of the features
    
    hidden_neurons: int
            This is the size the of layers in between the input and output layers

    output_neurons: int
            This is the size of the output neurons

    epochs: int
            This is the amount of times the model will iterate over the ENTIRE dataset.

    Returns
    -------
    model : nn.Module
            This is the model of the trained network
    
    y_pred: numpy array
            These are the predictions made.
    """
    
    best_params = {"input_neurons":input_neurons[0], "hidden_neurons": hidden_neurons[0],\
                    "output_neurons":output_neurons[0], "epochs":epochs[0]}
    best_recall = 0
    best_report = None
    for i in input_neurons:
        for k in hidden_neurons:
            for j in output_neurons:
                for l in epochs:
                    print("Parameters: Input_neurons: {}, Hidden_neurons: {}, Output_neurons: {}, Epochs: {}".format(\
                          i, k, j, l))
                    model, y_pred, _ = run_pytorch_pipeline(model_class, X_train,y_train, X_test, y_test,\
                                        input_neurons=i, output_neurons=j,\
                                        hidden_neurons=k, epochs=l, training_verbose=verbose)

                    report = report2dict(classification_report(y_test, y_pred))
                    
                    micro_recall = report[" micro avg"]["recall"]

                    if micro_recall > best_recall:
                        best_recall = micro_recall
                        best_report = report
                        best_params["input_neurons"] = i
                        best_params["hidden_neurons"] = k
                        best_params["output_neurons"] = j
                        best_params["epochs"] = l

    return best_params, best_report, y_pred

def get_class_counts(arr, train=True):

    if train:
        dataset = "Train"
    else:
        dataset = "Test"
        
    labels, counts = np.unique(arr, return_counts=True)
    print("Dataset {}".format(dataset))
    for index, label in np.ndenumerate(labels):
        print("Class = {}, count = {}".format(label, counts[index]))


def classification_table(df_test, y_test, svm_pred, nn_pred, nn_scores, count_above=2,\
                        score_threshold=0.02):

    """
    This method creates the classification table composing of the predicted values of
    the SVM and NN.  This table gives insight on which annotation might be similar to another
    class.  This would be helpful in determining better groupings.

    Parameters
    ----------
    df_test: dataframe
            This is the dataframe of the test set, we only care about
            the test set since that is the only one with labels that
            have been dealt with.

    y_test : numpy array
            This is the ground truth labeling of the X_test

    svm_pred: numpy array
            This is the array holding the predictions from the SVM
            
    nn_pred: numpy array
            This is the array holding the predictions from the Neural Network

    nn_scores: numpy array
            This is the array holding the softmax score per sample tested
    
    count_above: int
            This is the amount of max values from each row, where the differences will be 
            taken and checked against th score threshold.  If the difference is above the score threshold
            then that means those classes were not that similar on that image, else they are.  However,
            there is a caveat to this.  Right now the code is hardcoded for 2 and only 2 classes.  If more values
            are to be gathered then one must write more code to do that.

    score_threshold: float
            This is the threshold the difference between the top 2 scores must beat to be true.

    Returns
    -------
    df : dataframe
            This is the classification table.  

    """
    df = pd.DataFrame(index=df_test.index)
    # df["image_name"] = df_test["image_name"]
    df["ground_truth"] = y_test
    df["svm_preds"] = svm_pred
    df["nn_preds"] = nn_pred

    for i in range(nn_scores.shape[1]):
        df["nn_score_{}".format(i)] = nn_scores[:, i]
    
    boolean_list = []
    for i in range(nn_scores.shape[0]):
        arr_temp = nn_scores[i]
        indices = np.argsort(-np.asarray(arr_temp))[:count_above] # top 2 scores, even though this is a variable, it should stay at 2
                                                                # unless one plans to take the difference of more than two points
                                                                # which regardless would require better handling than what I have made.
        diff = abs(arr_temp[indices[0]]-arr_temp[indices[1]]) # gets the difference between the top two scores.
        boolean_list.append(diff < score_threshold) # stores the boolean in to a list
                                                    # BTW less means that they are similar do not do greater than
   
    df["top_{}_difference_under_{}".format(count_above, score_threshold)] = boolean_list

    # df.columns = ["image_name", "ground_truth", "SVM_preds", "NN1_preds", "NN1_score_1",\
    #  "NN1_score_2", "NN1_score_3", "NN1_score_4", "NN1_score_5", "Top_2_above_" + score_threshold]

    return df

def intersection_table(df, svm_pred, nn_pred):

    """
    This method is the driving method for training/testing a model and saving the report.
    This however does not save the model.

    Parameters
    ----------
    df_test: dataframe
            This is the dataframe of the test set, we only care about
            the test set since that is the only one with labels that
            have been dealt with.

    y_test : numpy array
            This is the ground truth labeling of the X_test

    svm_pred: numpy array
            This is the array holding the predictions from the SVM
            
    nn_pred: numpy array
            This is the array holding the predictions from the Neural Network

    nn_scores: numpy array
            This is the array holding the softmax score per sample tested
    
    count_above: int
            This is the amount of max values from each row, where the differences will be 
            taken and checked against th score threshold.  If the difference is above the score threshold
            then that means those classes were not that similar on that image, else they are.  However,
            there is a caveat to this.  Right now the code is hardcoded for 2 and only 2 classes.  If more values
            are to be gathered then one must write more code to do that.

    score_threshold: float
            This is the threshold the difference between the top 2 scores must beat to be true.

    Returns
    -------
    df : dataframe
            This is the classification table.  

    """
    labels = df["y_values"].unique()
    df["svm_preds"] = svm_pred
    df["nn_preds"] = nn_pred
    
    intersection_df = pd.DataFrame()
    
    intersection_df["class"] = labels

    # SVM counts where predictions that are the same as the ground truth and
    # this is per class.  So label is the class that the dataframe is looking
    # at and then the next bit of logic is looking to see where the 
    # predictions are the same as the ground truth
    counts = []
    for label in labels:
        
        count = df.loc[(df["y_values"] == label) & (df["y_values"] == df["svm_preds"]),"y_values"].count()
        counts.append(count)
        
    intersection_df["svm_correct"] = counts
    
    # One layer Neural Network counts where predictions that are the same as
    # the ground truth and this is per class.  So label is the class that 
    # the dataframe is looking at and then the next bit of logic is looking 
    # to see where the predictions are the same as the ground truth
    counts = []
    for label in labels:
        
        count = df.loc[(df["y_values"] == label) & (df["y_values"] == df["nn_preds"]),"y_values"].count()
        counts.append(count)
        
    intersection_df["nn_correct"] = counts
    
    # SVM and NN correct intersection
    counts = []
    for label in labels:
        
        count = df.loc[(df["y_values"] == label) & (df["y_values"] == df["nn_preds"]) & (df["y_values"] == df["svm_preds"]) & (df["svm_preds"] == df["nn_preds"]),"y_values"].count()
        counts.append(count)
        
    intersection_df["correct_intersection"] = counts
        
    # SVM counts where predictions that are NOT the same as the ground truth,
    # this is per class.  So label is the class that the dataframe is looking
    # at and then the next bit of logic is looking to see where the 
    # predictions are the same as the ground truth
    counts = []
    for label in labels:
        
        count = df.loc[(df["y_values"] == label) & (df["y_values"] != df["svm_preds"]),"y_values"].count()
        counts.append(count)
        
    intersection_df["svm_incorrect"] = counts
    
    # Neural Network counts where predictions that are the same as
    # the ground truth and this is per class.  So label is the class that 
    # the dataframe is looking at and then the next bit of logic is looking 
    # to see where the predictions are the same as the ground truth
    counts = []
    for label in labels:
        
        count = df.loc[(df["y_values"] == label) & (df["y_values"] != df["nn_preds"]),"y_values"].count()
        counts.append(count)
        
    intersection_df["nn_incorrect"] = counts
    
    # SVM and NN incorrect intersection
    counts = []
    for label in labels:
        
        count = df.loc[(df["y_values"] == label) & (df["y_values"] != df["nn_preds"]) & (df["y_values"] != df["svm_preds"]) & (df["svm_preds"] == df["nn_preds"]),"y_values"].count()
        counts.append(count)
        
    intersection_df["incorrect_intersection"] = counts
    intersection_df = intersection_df.set_index("class")
    
    return intersection_df

def visualize(df, image_path, color=(0,0,255)):
    
    images = df['image_name']
    

    for image in images:
        path = os.path.join(image_path, image )
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        temp_df = df.loc[df['image_name'] == image]
        for index, row in temp_df.iterrows():
            box = row['bbox']

            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(255,255,0), thickness=1)
            class_string = "Class ID= {}".format(row['y_values'])
            cv2.putText(img, class_string, (int(box[0]), int(box[1])),cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            print(class_string)
        print(path)

        cv2.imshow("visualizing", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def remove_duplicates_2d(X, y):

    x = np.random.rand(X.shape[1])
    y = X.dot(x)
    uniques, index = np.unique(y, return_index=True)
    arr = X[index]
    arr2 = y[index]

    return arr, arr2