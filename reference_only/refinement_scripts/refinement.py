from cvjson.cvj import CVJ
from cvjson.extensions.visualizer import Visualizer
import argparse
import operator
import os
import math
import numpy as np
import cv2 as cv
import merger
import datetime

# Not Working
# def find_three_anns_closest_to_center(ann_list, center_of_image):

#     similarity_list = []
#     min_list = []
#     for ann in ann_list:
#         bbox = ann["bbox"]

#         bbox_center = [(bbox[0] + (bbox[2] //2)), (bbox[1] + (bbox[3]//2))]

#         x_diff = (center_of_image[0] - bbox_center[0])**2
#         y_diff = (center_of_image[1] - bbox_center[1])**2

#         similarity_list.append(math.sqrt(x_diff + y_diff))
    
#     for i in range(3):
#         ## Trying to the the 3 minimum distances from the center of the image
#         ## Some bounding boxes are large and far away so just using area is not a good idea.
#         min_index, min_value = min(enumerate(similarity_list), key=operator.itemgetter(1))
#         min_list.append(min_index)

    
    # # Returning the three closest annotations
    # new_list = [ann_list[index] for index in min_list]
    # return new_list

def mask_to_poly(min_ann):
    im2, contours, hierarchy = cv2.findContours(np.asarray(min_ann["segmentation"]).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_ann["segmentation"] = contours
    return min_ann

def get_biggest_area_bbox(cvj_merged):

    image_id_2_anns = cvj_merged.get_image_id_2_anns()

    annotations = []
    area_list = []

    for id, anns in image_id_2_anns.items():
        for ann in anns:
            area_list.append(ann["area"])
    
        max_index, max_value = max(enumerate(area_list), key=operator.itemgetter(1))
        bbox = anns[max_index]["bbox"]
        class_id = anns[max_index]["category_id"]
        image_id = anns[max_index]["image_id"]
        id = anns[max_index]["id"]
        segm = anns[max_index]["segmentation"]

        ann = cvj_merged.entry_bbox(bbox, class_id, image_id, id)
        ann["segmentation"] = segm

        annotations.append(ann)

    cvj_merged["annotations"] = annotations

    return cvj_merged

def parse_args():
    parser = argparse.ArgumentParser(description='Refine Detectron boats')

    parser.add_argument('--image_dir', required=True, help='To merge')

    parser.add_argument('--inferred_json', required=True,help='The bad annotations file that was inferred to make')
    parser.add_argument("--original_json", required=True, help="The original file to get the classes from.")

    parser.add_argument("--vis", required=False, help="set this if you want to visualize and make sure you like the annotations", action="store_true")
    parser.add_argument("--refine", required=False, action="store_true", help="This is used to know if refinement should happen.")

    args = parser.parse_args()

    return args


def refine(cvj_merged, refinement_function):

    cvj_merged = refinement_function(cvj_merged)

    return cvj_merged

def main():

    args = parse_args()
    cvj_infer = CVJ(args.inferred_json, args.image_dir)
    cvj_origin = CVJ(args.original_json, args.image_dir)

    cvj_obj = merger.merge(cvj_infer, cvj_origin)

    if args.refine:
        cvj_obj = refine(cvj_obj, get_biggest_area_bbox) # to be replaced with something more generic when integrated with CVJSON

    # Creating the Unique name
    dir_name = os.path.dirname(args.inferred_json)
    date, time = str(datetime.datetime.now()).split(" ")
    time = time.split('.')[0].replace(':', '_')
    name = "refined_" + date + '_' + time + ".json"
    path = os.path.join(dir_name, name)

    # saving the merged and/or refined file
    cvj_obj.save_internal_json(save_name=path)

    if args.vis:
        Visualizer(cvj_obj).visualize_bboxes()

if __name__ == "__main__":
    main()
        



