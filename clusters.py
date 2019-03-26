from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import pandas as pd

from cvjson.cvj import CVJ
from experiment_methods import get_data
import os
import plotting


def kmeans_cluster_analysis(hdf5_path, cvj_obj, file_name=None):
    """
    This method clusters the data found in the hdf5_path and then clusters them using Kmeans.
    This method takes the clusters and the classes and figures out which cluster each class has
    been most assigned to.  This is done 10 times and then stored in a DataFrame object using 
    pandas.

    The results of this method are only interesting if one looks at how frequently the classes
    get assigned to the same cluster.  This tells us that there is similarity between those classes
    if they are consistently assigned to the same cluster.

    Parameters
    ----------
    hdf5_path: DataFrame object
            This dataframe object comes from TwoD_data()

    cvj_obj: CVJ object
            This a CVJ object from the CVJSON library.  This most likely should be filled with
            the data from the training set.  However, this cvj object is only used to convert
            the ids to their respective names.

    file_name: str
            This the file name/file path where the file should be saved.

    Returns
    -------
    df: DataFrame object
            This dataframe holds all of the information regarding how the classes were being clustered.

    """

    X, y, labels, counts = get_data(hdf5_path)

    # Clustering and inspecting
    pseudo_df = []
    for i in range(10):
        model = KMeans()
        clusters = model.fit(X).labels_
        values = np.unique(clusters)

        df = pd.DataFrame(y, index=None)
        df.columns = ["y_values"]
        df["cluster"] = clusters
        counts = df.groupby("y_values")["cluster"].value_counts()

        viewed_indices = []

        most_index_cluster = []

        for k in counts.index.tolist():
            if k[0] not in viewed_indices:
                viewed_indices.append(k[0])
                most_index_cluster.append(k[1])
        pseudo_df.append(most_index_cluster)

    df = pd.DataFrame(pseudo_df)
    df = df.transpose()
    df["class"] = [cvj_obj.get_class_id_2_name(i) for i in viewed_indices]

    if file_name == None:
        file_name = "kmeans_cluster_analysis"
    df.to_csv(file_name + ".csv")
    print(df)

    return df


























### Threshold is for the total amount of annotations a certain combination has.
# threshold = 50

# output_dir = "/home/ben/Desktop/mapping_labels/combination_csvs"

# values, value_counts = np.unique(y, return_counts=True)

# name_2_count = {}
# for index, i in enumerate(values):
#     if value_counts[index] < threshold:
#         continue
#     name_2_count[train_cvj.get_class_id_2_name(i)] = [value_counts[index]]



# name_2_count = pd.DataFrame.from_dict(name_2_count)

# cats = list(name_2_count.keys())




# list_of_combinations = []
# for i in range(2, 6):
#     combinations = list(set(itertools.combinations(cats, i)))
    
#     for i in combinations:
#         df = name_2_count[list(i)]
#         sum_ = df.loc[0, :].sum()
#         if sum_ < threshold:
#             continue
#         else:
#             list_of_combinations.append(list(i))

# for i in list_of_combinations:
#     df = name_2_count[i]
#     df["sum"] = df.loc[0, :].sum()
#     df.to_csv(os.path.join(output_dir, '_'.join(i) + ".csv"))



# model = TSNE(n_components=3)
# tsne_matrix = model.fit_transform(X)

# for i in list_of_combinations
# ids = list(train_cvj.get_class_id_2_name().keys()) 

# print(ids[:3])
# print(y.shape)
# print(np.unique(y))

# hmm = y[np.isin(y, ids[:3])]

# print(ids[:3])
# print(hmm.shape)
# print(np.unique(hmm))