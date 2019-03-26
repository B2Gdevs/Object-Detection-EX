import h5py
from cvjson.cvj import CVJ
from cvjson.extensions.visualizer import Visualizer
import cv2
import os
import json
import operator
import hdf5_utils
import numpy as np
import torch.nn as nn
import torch 


def bb_intersection_over_union(boxA, boxB):
    """
    This method gets the IoU score for the two boxes given.  Boxes must be in
    x0,y0,x1,y1 format.  This means that at indices 2 and 3 are x and y coordinates that
    are the higher values.  They are not the height and width of the box.

    Parameters
    ----------
    boxA: list
            This is a bounding box in the form of [x0,y0,x1,y1]

    boxb: list
            This is a bounding box in the form of [x0,y0,x1,y1]

    Returns
    -------
    iou: float
            This is the amount of overlap the bounding boxes have with their combined total 
            area.
    """
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


# If you are thinking about using this on the MARDCT test set, don't use the one found in
# the testset_mardct_map.py file.  The reason is that the testset didn't have all of the class names
# there for it is better to use the training set to find the class names. 
#
# The method below, has only the ground truth for whatever set it is and the categories.
# However, the mardct test set needs the categories from the training CVJ object and the annotations
# from the test CVJ object.  So this requires a different method.
def get_features_to_GT(file_path, cvj_obj, output_dir, image_path=None):
    """
    As object detection algorithms are showing to be non-deterministic, this method 
    gets the closest bounding boxes to the ground truth labels supplied in the CVJ object.  This is done
    by using IoU.  The best IoUs get the labels.
    The bounding boxes that will be kept are the ones inside the HDF5 file.  This is because it doesn't make 
    sense to store the features of the ground truth labels when they are truly the features of the ground truth labels.


    Parameters
    ----------
    file_path: string
            path to hdf5 file

    cvj_obj: CVJ object
            is the CVJ object that has the annotations you want the features too.
    
    output_dir: string
            the path to the directory you want the newe HDF5 file to be with the annotations, labels, and features.

    image_path: string
            path to images in cvj_obj file.  If this is not None then it will visualize the bounding boxes.

    """

    file_ = h5py.File(file_path, 'r')
    mardct_cats = list(cvj_obj.get_category_names()) # this grabs all of the category names and stores them in a list.

    base_name = os.path.splitext(os.path.basename(file_path))[0] + "_mapped_features.hdf5"
    output_name = os.path.join(output_dir, base_name )

    feature_file = hdf5_utils.create_datastore(output_name, "image_directory", mardct_cats)

    near_negative_name = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + "_near_negatives.hdf5")
    near_negative_file = hdf5_utils.create_datastore(near_negative_name, "image_directory", mardct_cats) # Dr. Tesic wants to maintain the annotations 

    for image, values in list(file_["frames"].items()):

        img_id = cvj_obj.get_filename_2_image_id(image)
        anns = cvj_obj.get_image_id_2_anns(img_id)

        boxes = values["bounding_boxes"]
        features = values["descriptors"]
        scores = values["scores"]

        anns_no_features = [] # to store annotations from the ground truth labels that didn't have a corresponding detection.
        
        #### For visualization only.
        if image_path is not None:
            file_path = os.path.join(image_path, image)
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        ####

        # set up the dictionary to house all detections found in the ground truth and the inference
        raw_dict = {}
        raw_dict["boxes"] = []
        raw_dict["scores"] = []
        raw_dict["descriptors"] = []
        raw_dict["category"] = []
        for ann in anns:
            json_box = ann["bbox"]
            json_xyxy_box = [json_box[0], json_box[1], (json_box[0] + json_box[2]), (json_box[1] + json_box[3])] # converting x,y,h,w format to x0,y0,x1,y1
            iou_list = []

            if len(boxes) == 0:
                anns_no_features.append(ann) # ok I don't know what I am going to do with this...This is to save the annotations that didn't have there features detected.
                continue

            print("length of boxes = {}".format(len(boxes))) # I have this here so I can tell if it makes sense when I get a print message that says there are
                                                             # no more detections, but might have more ground truth annotations.
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

                #### For visualization only.
                if image_path is not None:
                    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,0), 1)

                    cv2.rectangle(img, (json_xyxy_box[0], json_xyxy_box[1]), (json_xyxy_box[2], json_xyxy_box[3]), (0,255,255), 1)

                    class_string = "Class = {}, Class ID= {}".format(cvj_obj.get_class_id_2_name(cat), cat)
                    cv2.putText(img, class_string, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3)
                ####

                # This is needed to reduce the boxes on the image to check for IoU again.  Some boxes overlap
                # which means they might have a better IoU with another detection.  So just getting rid of the best one each
                # loop solves that problem.
                np.delete(boxes, feature_index)
            else:
                print("value == 0") # I want to know when there are more ground truth annotations than detections.  This signals that.
                anns_no_features.append(ann)
        
        #### For Visulization only.
        if image_path is not None:
            cv2.imshow("test", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            ####

        raw_dict["category"] = np.asarray(raw_dict["category"])
        raw_dict["descriptors"] = np.asarray(raw_dict["descriptors"])
        raw_dict["boxes"] = np.asarray(raw_dict["boxes"])
        raw_dict["scores"] = np.asarray(raw_dict["scores"])

        feature_file = hdf5_utils.add_frame(feature_file, image, raw_dict)
        
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

    string =\
    """
    train_image_path = "/home/ben/Desktop/M12_Folder_Mimic/Datasets/Train/Images/Mardct/completed_train_refinement"
    train_cvj = CVJ("/home/ben/Desktop/M12_Folder_Mimic/Datasets/Train/Cocoized/without_coco_categories/completed_train_refinement.json", train_image_path)

    hdf_file_path = "/home/ben/Desktop/outputs/boat.hdf5"
    output_dir = "/home/ben/Desktop"


    get_features_to_GT(hdf_file_path, train_cvj, output_dir)
    """

    print("Below is how one can use the code found in this script.\n"\
        "It is recommended to call these methods in another script"\
        " for readability's sake.")
    
    print(string)




