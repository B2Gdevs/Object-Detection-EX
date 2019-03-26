"""
Author: Benjamin Garrard

This script is needed to be used with the test set of mardct test annotation. 
This is because only 19 classes were found in the test set after refinement and there actually needs
to be 24.  This script handles the "completed_test_refinement.json" annotation file only.

For every other time and dataset you want to map features from detectron to, use "map_labels.py"
"""

import h5py
from cvjson.cvj import CVJ
from cvjson.extensions.visualizer import Visualizer
import cv2
import os
import json
import operator
import hdf5_utils
import numpy as np

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

def save_json(json_data, name):
    with open(name, 'w') as file:
        json.dump(json_data, file)


# Please for the love of god use this method if you are trying to map the testset labels
# to the predicted labels given by Detectron.
#
#
def get_features_to_GT(file_path, image_path, cvj_obj, train_cvj, output_dir): 

    file_ = h5py.File(file_path, 'r')
    mardct_cats = list(train_cvj.get_category_names()) # this will need to be more generic later
    # print(len(mardct_cats))
    # input()

    base_name = os.path.splitext(os.path.basename(file_path))[0] + "_mapped_features.hdf5"
    output_name = os.path.join(output_dir, base_name )

    feature_file = hdf5_utils.create_datastore(output_name, "image_directory", mardct_cats)

    near_negative_name = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + "near_negatives.hdf5")
    near_negative_file = hdf5_utils.create_datastore(near_negative_name, "image_directory", mardct_cats) # Dr. Tesic wants to maintain the annotations 

    for image, values in list(file_["frames"].items()):

        img_id = cvj_obj.get_filename_2_image_id(image)
        anns = cvj_obj.get_image_id_2_anns(img_id)

        boxes = values["bounding_boxes"]

        features = values["descriptors"]
        scores = values["scores"]
        # file_path = os.path.join(image_path, image)

        anns_no_features = []
        
        # set up the dictionary to house all detections found in the ground truth and the inference
        raw_dict = {}
        raw_dict["boxes"] = []
        raw_dict["scores"] = []
        raw_dict["descriptors"] = []
        raw_dict["category"] = []
        # img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        for ann in anns:
            json_box = ann["bbox"]
            json_xyxy_box = [json_box[0], json_box[1], (json_box[0] + json_box[2]), (json_box[1] + json_box[3])] # converting x,y,h,w format to x0,y0,x1,y1
            iou_list = []

            if len(boxes) == 0:
                anns_no_features.append(ann) # ok I don't know what I am going to do with this...This is to save the annotations that didn't have there features detected.
                continue

            print("length of boxes = {}".format(len(boxes)))
            for i, box in enumerate(boxes):
                if len(box) > 0:
                    
                    iou_list.append(bb_intersection_over_union(json_xyxy_box, box)) # getting the IoU for every box and comparing it to the GT

            if len(iou_list) != 0:
                feature_index, val = max(enumerate(iou_list), key=operator.itemgetter(1)) # getting the best IoU threshold and its index
            else:
                val = 0
                        
            if val != 0:
                cat = int(ann["category_id"]) # We need the label from the GT for the detection that best fits the GT.
                raw_dict["descriptors"].append(np.copy(features[feature_index]))
                raw_dict["boxes"].append(np.copy(boxes[feature_index]))
                raw_dict["scores"].append(np.copy(scores[feature_index]))
                raw_dict["category"].append(np.asarray(cat))
                box = boxes[feature_index]
                # cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,0), 1)

                # cv2.rectangle(img, (json_xyxy_box[0], json_xyxy_box[1]), (json_xyxy_box[2], json_xyxy_box[3]), (0,255,255), 1)

                # class_string = "Class = {}, Class ID= {}".format(train_cvj.get_class_id_2_name(cat), cat)
                # cv2.putText(img, class_string, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3)
                np.delete(boxes, feature_index)
            else:
                print("value == 0")
                anns_no_features.append(ann)

        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        raw_dict["category"] = np.asarray(raw_dict["category"])
        raw_dict["descriptors"] = np.asarray(raw_dict["descriptors"])
        raw_dict["boxes"] = np.asarray(raw_dict["boxes"])
        raw_dict["scores"] = np.asarray(raw_dict["scores"])

        feature_file = hdf5_utils.add_frame(feature_file, image, raw_dict)
        
        # print(list(feature_file["frames"].keys()))
        frame = feature_file["frames"][image]

        ds = frame.create_dataset("categories",raw_dict['category'].shape)
        ds[...] = raw_dict['category']

        # save detections that were made, but not found in the ground truth file.
        if len(boxes) != 0:
            detections_not_found_in_ground_truth = {}

            detections_not_found_in_ground_truth["boxes"] = []
            detections_not_found_in_ground_truth["scores"] = []
            detections_not_found_in_ground_truth["descriptors"] = []

            for i, box in enumerate(boxes):
                detections_not_found_in_ground_truth["boxes"].append(box)
                detections_not_found_in_ground_truth["scores"].append(scores[i])
                detections_not_found_in_ground_truth["descriptors"].append(features[i])

            detections_not_found_in_ground_truth["descriptors"] = np.asarray(detections_not_found_in_ground_truth["descriptors"])
            detections_not_found_in_ground_truth["boxes"] = np.asarray(detections_not_found_in_ground_truth["boxes"])
            detections_not_found_in_ground_truth["scores"] = np.asarray(detections_not_found_in_ground_truth["scores"])

            hdf5_utils.add_frame(near_negative_file, image, detections_not_found_in_ground_truth)

    near_negative_file.close()
    feature_file.close()

    cvj_obj["annotations"] = anns_no_features

    file_name_for_gt_that_need_features = os.path.splitext(os.path.basename(cvj_obj.json_path))[0] + "_need_features.json" 
    cvj_obj.save_internal_json(os.path.join(output_dir, file_name_for_gt_that_need_features))


if __name__ == "__main__":

    string = \
    """
    test_image_path = "/home/ben/Desktop/M12_Folder_Mimic/Datasets/Validation/Images/completed_refinement_test_images"
    test_cvj = CVJ("/home/ben/Desktop/M12_Folder_Mimic/Datasets/Validation/Cocoized/without_coco_categories/completed_test_refinement.json", test_image_path )

    train_image_path = "/home/ben/Desktop/M12_Folder_Mimic/Datasets/Train/Images/Mardct/completed_train_refinement"
    train_cvj = CVJ("/home/ben/Desktop/M12_Folder_Mimic/Datasets/Train/Cocoized/without_coco_categories/completed_train_refinement.json", train_image_path)

    hdf_file_path = "/home/ben/Desktop/outputs/boat.hdf5"
    output_dir = "/home/ben/Desktop"

    get_features_to_GT(hdf_file_path, test_image_path, test_cvj, train_cvj, output_dir)
    """

    print("Below is how one can use the code found in this script.\n"\
        "It is recommended to call these methods in another script"\
        " for readability's sake.")

    print(string)

