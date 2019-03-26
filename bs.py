#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:05:51 2018

@author: ben
"""
import pandas as pd
import numpy as np
import os

df = pd.read_csv("/home/ben/Desktop/mapping_labels/classification_table.csv")

duplicates = df[df.duplicated()]

new_df = df.loc[df['image_name']=='0.jpg']

# =============================================================================
# nn_scores = df.loc[3, 'nn_score_0':'nn_score_4']
# arr_temp = nn_scores
# indices = np.argsort(-np.asarray(arr_temp))[:2] 
#                                                         
# score1 = nn_scores[indices[0]]
# score2 = nn_scores[indices[1]]
# diff = abs(score1-score2)
# 
# print(diff <0.02)
# 
# =============================================================================

from cvjson.cvj import CVJ
from cvjson.extensions.visualizer import Visualizer

train_json_path = "/home/ben/Desktop/completed_train_refinement.json"
train_image_path = "/home/ben/Desktop/zvezda_data/mardct/Completed_mardct_refinement/completed_train_refinement"

test_json_path = "/home/ben/Desktop/completed_test_refinement.json"
test_image_path = "/home/ben/Desktop/zvezda_data/mardct/Completed_mardct_refinement/completed_refinement_test_images"

# train_cvj = CVJ(train_json_path, train_image_path)
test_cvj = CVJ(test_json_path, test_image_path)

image_names = duplicates['image_name']
Visualizer(test_cvj).view_images_by_name(image_names)


import numpy as np

data = np.array([1,3,3,4],
                 [1,8,9,4],
                 [1,8,3,3,4])

x = np.random.rand(data.shape[1])
y = data.dot(x)


uniques, index = np.unique(y, return_index=True)
arr = data[index]


a= ["aa","bb",'cc']
b= ["aa","dd", 'kk']
c= np.asarray([a,b])

to_remove = ['aa', 'kk']

c = [x[~np.isin(x,to_remove)].tolist() for x in c]

print(c)

import pandas as pd

data = pd.DataFrame(data)

