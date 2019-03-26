from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import h5py
import numpy as np
import matplotlib as plt
from cvjson.cvj import CVJ
import os
import copy
from pyfiglet import Figlet
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def get_data(hdf5_path, list_of_classes_to_keep=None, semantics=False, get_counts=False, negatives=False):

    X = []
    Y = []
    file_ = h5py.File(hdf5_path)

    for image, values in list(file_["frames"].items()):

        # boxes = values["bounding_boxes"]
        features = values["descriptors"]
        features = np.asarray(features)
        scores = values["scores"]

        try: 
            features = features.reshape(scores.shape[0], -1)

        except ValueError:
            print(features.shape)
            print(features)

        if negatives == False:
            categories = values["categories"]
            categories = np.asarray(categories).astype(int)
        # else:
        #     categories = np.asarray([5000 for i in range(features.shape[0])]) # 5000 is the id I am using for the negative class
        #     list_of_classes_to_keep = [5000]
        
        ##### START#### This iterates through and makes the X and Y data
        if len(X) == 0:
            if semantics:
                pass
            if list_of_classes_to_keep != None : 
                for index, cat in enumerate(categories):
                    if cat in list_of_classes_to_keep:
                        X.append(np.copy(features[index]))
                        Y.append(np.copy(categories[index]))
            else:
                X = np.copy(features)
                Y = np.copy(categories)
            
            X = np.asarray(X)
            Y = np.asarray(Y)
        else:
            if list_of_classes_to_keep != None:

                for index, cat in enumerate(categories):
                    if cat in list_of_classes_to_keep:
                        Y = np.vstack((Y, cat))
                        X = np.vstack((X, features[index]))

                        if semantics:
                            X = np.concatenate(features[index], np.asarray(scores[index]))

            else:
                for index, cat in enumerate(categories):
                    Y = np.vstack((Y, cat))
                    X = np.vstack((X, features[index]))
        #### END ####
    
    print(X.shape)
    print(Y.shape)
    Y = Y.ravel() # Flattening to a 1-d array
    category_names = np.asarray(file_["class_names"])
    file_.close()
    return X, Y, category_names


def run_classifier(classifier, train_path=None, test_path=None, training_negatives=None, test_negatives=None, train_cvj=None, test_cvj=None, n_best_class=6, test_path_is_negative=False):

    if train_cvj != None:
        list_of_class_to_keep = get_top_count_ids(test_cvj, train_cvj, n_best_class)


    train_positives_count = 0
    train_negatives_count = 0
    if train_path != None:
        X_train, Y_train, categories = get_data(train_path, list_of_classes_to_keep=list_of_class_to_keep)
        train_positives_count = X_train.shape[0] 
        if training_negatives != None:
            for file_path in training_negatives:
                try:
                    X_neg, Y_neg, _ = get_data(file_path, list_of_classes_to_keep=list_of_class_to_keep, negatives=True)
                    X_train = np.concatenate((X_train, X_neg))
                    Y_train = np.concatenate((Y_train, Y_neg))
                except ValueError:
                    continue
            train_negatives_count = X_train.shape[0] - train_positives_count

        classifer.fit(X_train, Y_train)
        print("\n\nTraining Positive Count = {}, Training Negative Count {}".format(train_positives_count, train_negatives_count))
        
        

    test_positives_count = 0
    test_negatives_count = 0
    if test_path != None and test_path_is_negative != True:
        X_test, Y_test, _ = get_data(test_path, list_of_classes_to_keep=list_of_class_to_keep)
        test_positives_count = X_test.shape[0]

        if test_negatives != None:
            for file_path in test_negatives:
                try:
                    X_neg, Y_neg, _ = get_data(file_path, list_of_classes_to_keep=list_of_class_to_keep, negatives=True)
                    X_test = np.concatenate((X_test, X_neg))
                    Y_test = np.concatenate((Y_test, Y_neg))
                except ValueError:
                    continue
            test_negatives_count = test_positives_count - X_test.shape[0]

    
    elif test_path_is_negative:
        X_test, Y_test, _ = get_data(test_path, list_of_classes_to_keep=list_of_class_to_keep, negatives=True)
        test_positives_count = 0
        test_negatives_count = X_test.shape[0]

    else:
        print("No test files given, returning classifier.")
        return classifier

    preds = classifer.predict(X_test)
    print("Test Positive Count = {}, Test Negative Count {}".format(test_positives_count, test_negatives_count))
        
    if train_cvj != None:
        list_of_names = [train_cvj.get_class_id_2_name(class_id) for class_id in list_of_class_to_keep]
    else:
        list_of_class_to_keep = [id for id in range(len(categories) + 1) if id != 5] # 5 is MARDCT specific, remove if statement if is using a different dataset
        list_of_names = categories

    # list_of_names.append("negative")
    # list_of_class_to_keep.append(5000) # this is what I am using for a negative ID

    print(classification_report( Y_test, preds, target_names=list_of_names, labels=list_of_class_to_keep))
    print("\n\naccuracy = {}".format(accuracy_score(Y_test, preds, normalize=True)))
    cm = confusion_matrix(Y_test, preds, labels=list_of_class_to_keep)
    # cm_heatmap(cm, list_of_names)
    print_cm(cm, list_of_names)

    return classifer, np.asarray(preds), cm, list_of_names



def get_top_count_ids(test_cvj_obj, train_cvj_obj, top_count):
    class_2_anns = copy.deepcopy(test_cvj_obj.get_class_id_2_anns())

    list_of_classes = []
    for i in range(top_count):
        key = max(class_2_anns, key= lambda x: len(class_2_anns[x]))
        # print(key)
        # print(train_cvj_obj.get_class_id_2_name(key))
        list_of_classes.append(key)
        del class_2_anns[key]

    return list_of_classes

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
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
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

def cm_heatmap(cm, labels, file_name, show=False):
    df_cm = pd.DataFrame(cm, index = [i for i in labels],
                  columns = [i for i in labels])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt="d")

    if show:
        plt.show()
        input()
    plt.savefig(file_name)

####### PATHS
train_labeled_features_path = "/home/ben/Desktop/mapped_features/train_mardct_coco_fine_ipatch/boat_mapped_features.hdf5" 
test_labeled_features_path = "/home/ben/Desktop/mapped_features/test_mardct_coco_fine_ipatch/boat_mapped_features.hdf5"


train_image_path = "/home/ben/Desktop/M12_Folder_Mimic/Datasets/Train/Images/Mardct/completed_train_refinement"
train_cvj = CVJ("/home/ben/Desktop/M12_Folder_Mimic/Datasets/Train/Cocoized/without_coco_categories/completed_train_refinement.json", train_image_path)


test_image_path = "/home/ben/Desktop/M12_Folder_Mimic/Datasets/Validation/Images/completed_refinement_test_images"
test_cvj = CVJ("/home/ben/Desktop/M12_Folder_Mimic/Datasets/Validation/Cocoized/without_coco_categories/completed_test_refinement.json", test_image_path)

boat_near_negatives = "/home/ben/Desktop/mapped_features/train_mardct_coco_fine_ipatch/boat_near_negatives.hdf5" #boat specific near negatives from the boat.hdf5 file produced by detectron
train_negatives_from_detectron_dir = "/home/ben/Desktop/outputs/train" # this is the directory for every class detectron detected that was not a boat!!!! # Train
test_negatives_from_detectron_dir = "/home/ben/Desktop/outputs/test"
######## PATHS

# print text transform.  This is used to set the type of figlet font to use.  This will be helpful in identifying which results are from what.
figlet = Figlet(font="big")



#### MODEL TRAINED WITH ONLY POSITIVES BELOW!!!!!!!!!!!!!!!!!!!!!!#########################3


##### Start training model with no negatives, testset has no negatives

print("\n\n")
print(figlet.renderText("Testset No Negatives"))
print("\n\n Model is trained with only postive samples")

print("\n\n")
figlet.setFont(font="big")  
print(figlet.renderText("Begin")) # Making a START marker

classifer = SVC(kernel="rbf", degree=3, decision_function_shape="ovr", gamma="auto")

classifier, _, cm, labels = run_classifier(classifer, train_path=train_labeled_features_path, test_path=test_labeled_features_path, train_cvj=train_cvj, test_cvj=test_cvj)

file_name = "Train_no_negs_test_no_negs.png"

cm_heatmap(cm, labels, file_name)


print("\n\n")
print(figlet.renderText("End")) # Making a END marker
input()

#### End TRAIN Test, NO NEGATIVES


# # START test model trained with no negatives against a test set with positive and negatives
# print("\n\n")
# print(figlet.renderText("Testset Normal"))
# print("\n\n Model is trained with only postive samples")

# print("\n\n")
# figlet.setFont(font="big")  
# print(figlet.renderText("Begin")) # Making a START marker


# files = [os.path.join(test_negatives_from_detectron_dir, file_) for file_ in os.listdir(test_negatives_from_detectron_dir) if file_.endswith(".hdf5")]

# classifier, _, cm, labels = run_classifier(classifer, test_path=test_labeled_features_path, test_negatives = files, train_cvj=train_cvj, test_cvj=test_cvj)

# file_name = "Train_no_negs_test_normal.png"

# cm_heatmap(cm, labels, file_name)

# print("\n\n")
# print(figlet.renderText("End")) # Making a END marker
# # END test model trained with no negatives against a test set with negatives


# ### START testing with model trained with no negatives against a test set that only has negatives

# print("\n\n")
# print(figlet.renderText("Testset All Negatives"))
# print("\n\n Model is trained with only postive samples")

# print("\n\n")
# figlet.setFont(font="big")  
# print(figlet.renderText("Begin")) # Making a START marker

# files = [os.path.join(train_negatives_from_detectron_dir, file_) for file_ in os.listdir(train_negatives_from_detectron_dir) if file_.endswith(".hdf5")]

# print("\n\n ############################ File being read is {} and the results are below!. ######################\n\n".format(os.path.basename(boat_near_negatives)))
# classifier, predictions, cm, labels = run_classifier(classifer, test_path=boat_near_negatives, test_path_is_negative=True, train_cvj=train_cvj, test_cvj=test_cvj)

# counts = predictions
# saving_cm = cm

# print("\nThe results directly below this are from the BOAT hdf file.  This means these detections weren't found in the ground truth and thus are false positives.")
# values, value_counts = np.unique(counts, return_counts=True) # doing this before so we can get a read on the false positives of the BOATS hdf.  These are near positives rather than near negatives

# for index, class_id in np.ndenumerate(values):
#     if class_id != 5000: # My negative class id is 5000
#         print("Class: {}, FalsePositive_Count: {}".format(train_cvj.get_class_id_2_name(class_id), value_counts[index]))
#     else:
#         print("Class: {}, FalsePositive_Count: {}".format("Negative", value_counts[index]))

# for file_ in files:
#     # print(figlet.renderText("File being read is {}".format( os.path.basename(file_))))
#     print("\n\n ############################ File being read is {} and the results are below!. ######################\n\n".format(os.path.basename(file_)))
#     try:
#         classifier, predictions,cm,labels = run_classifier(classifer, test_path=file_, test_path_is_negative=True, train_cvj=train_cvj, test_cvj=test_cvj)
#         counts = np.concatenate((counts, predictions))
#         values, value_counts = np.unique(counts, return_counts=True)
#         saving_cm += cm
#         break
#         # end_total = np.asarray((values, value_counts)).T
#     except ValueError:
#         continue

# print()
# for index, class_id in np.ndenumerate(values):
#     if class_id != 5000: # My negative class id is 5000
#         print("Class: {}, FalsePositive_Count: {}".format(train_cvj.get_class_id_2_name(class_id), value_counts[index]))
#     else:
#         print("Class: {}, FalsePositive_Count: {}".format("Negative", value_counts[index]))

# file_name = "Train_no_negs_test_all_negs.png"

# cm_heatmap(saving_cm, labels, file_name)

# print("\n\n")
# print(figlet.renderText("End")) # Making a END marker
# #### End TEST with NEGATIVES





# #### MODEL TRAINED WITH POSITIVE AND NEGATIVES BELOW!!!!!!!!!!!!!!!!!!!!!!#########################3


# ##### Start Training a model with negatives, Testset has no Negatives ######
# print("\n\n")
# print(figlet.renderText("Testset No Negatives"))
# print("\n\n Model is trained with postive and negative samples")

# print("\n\n")
# figlet.setFont(font="big") 
# print(figlet.renderText("Begin")) # Making a START marker

# classifer = SVC(kernel="rbf", degree=3, decision_function_shape="ovr", gamma="auto")

# files = [os.path.join(train_negatives_from_detectron_dir, file_) for file_ in os.listdir(train_negatives_from_detectron_dir) if file_.endswith(".hdf5")]

# classifier, _, cm, labels = run_classifier(classifer, train_path=train_labeled_features_path, test_path=test_labeled_features_path, training_negatives = files, train_cvj=train_cvj, test_cvj=test_cvj)

# file_name = "Train_normal_test_no_negs.png"

# cm_heatmap(cm, labels, file_name)

# print("\n\n")
# print(figlet.renderText("End")) # Making a END marker
# ##### End Train with negatives ######



# ##### START testing model trained with positive and negatives on a normal test set with negatives
# print(figlet.renderText("Testset Normal"))
# print("\n\n Model is trained with postive and negative samples")

# print("\n\n")
# figlet.setFont(font="big")  
# print(figlet.renderText("Begin")) # Making a START marker

# files = [os.path.join(test_negatives_from_detectron_dir, file_) for file_ in os.listdir(test_negatives_from_detectron_dir) if file_.endswith(".hdf5")]

# classifier, _,cm, labels = run_classifier(classifer, test_path=test_labeled_features_path, test_negatives = files, train_cvj=train_cvj, test_cvj=test_cvj)

# file_name = "Train_normal_test_normal.png"

# cm_heatmap(cm, labels, file_name)

# print("\n\n")
# print(figlet.renderText("End")) # Making a END marker

# ##### END testing model trained with positive and negatives on a test set with negatives



# ##### START testing model trained with positive and negatives on a test set with only negatives

# print("\n\n")
# print(figlet.renderText("Testset All Negatives"))
# print("\n\n Model is trained with postive and negative samples")

# print("\n\n")
# figlet.setFont(font="big")  
# print(figlet.renderText("Begin")) # Making a START marker

# files = [os.path.join(train_negatives_from_detectron_dir, file_) for file_ in os.listdir(train_negatives_from_detectron_dir) if file_.endswith(".hdf5")]

# print("\n\n ############################ File being read is {} and the results are below!. ######################\n\n".format(os.path.basename(boat_near_negatives)))
# classifier, predictions, cm, labels = run_classifier(classifer, test_path=boat_near_negatives, test_path_is_negative=True, train_cvj=train_cvj, test_cvj=test_cvj)

# counts = predictions
# saving_cm = cm

# print("\nThe results directly below this are from the BOAT hdf file.  This means these detections weren't found in the ground truth and thus are false positives.")
# values, value_counts = np.unique(counts, return_counts=True) # doing this before so we can get a read on the false positives of the BOATS hdf.  These are near positives rather than near negatives

# for index, class_id in np.ndenumerate(values):
#     if class_id != 5000: # My negative class id is 5000
#         print("Class: {}, FalsePositive_Count: {}".format(train_cvj.get_class_id_2_name(class_id), value_counts[index]))
#     else:
#         print("Class: {}, FalsePositive_Count: {}".format("Negative", value_counts[index]))

# for file_ in files:
#     # print(figlet.renderText("File being read is {}".format( os.path.basename(file_))))
#     print("\n\n ############################ File being read is {} and the results are below!. ######################\n\n".format(os.path.basename(file_)))
#     try:
#         classifier, predictions = run_classifier(classifer, test_path=file_, test_path_is_negative=True, train_cvj=train_cvj, test_cvj=test_cvj)
#         counts = np.concatenate((counts, predictions))
#         values, value_counts = np.unique(counts, return_counts=True)
#         saving_cm += cm
#         break
#         # end_total = np.asarray((values, value_counts)).T
#     except ValueError:
#         continue

# print()
# for index, class_id in np.ndenumerate(values):
#     if class_id != 5000: # My negative class id is 5000
#         print("Class: {}, FalsePositive_Count: {}".format(train_cvj.get_class_id_2_name(class_id), value_counts[index]))
#     else:
#         print("Class: {}, FalsePositive_Count: {}".format("Negative", value_counts[index]))

# file_name = "Train_normal_test_all_negs.png"

# cm_heatmap(saving_cm, labels, file_name)

# print("\n\n")
# print(figlet.renderText("End")) # Making a END marker

# ##### END testing model trained with positive and negatives on a test set with only negatives


"""python svms.py  2>&1 | tee svm_results.txt"""# call with this when ready.