import numpy as np
from cvjson.cvj import CVJ
from cvjson.extensions.visualizer import Visualizer
import refinement
import os
import argparse
import datetime

def merge(cvj_infer, cvj_origin, show_errors=False):

    cvj_infer = find_extensions(cvj_infer)

    infer_file_2_id = cvj_infer.get_filename_2_image_id()
    infer_id_2_file = cvj_infer.get_image_id_2_filename()
    infer_image_id_2_anns = cvj_infer.get_image_id_2_anns()

    origin_file_2_id = cvj_origin.get_filename_2_image_id()
    origin_id_2_file = cvj_origin.get_image_id_2_filename()
    origin_image_id_2_anns = cvj_origin.get_image_id_2_anns()

    #there is only 1 class per image
    fname_2_class = {}

    # get classes from origin since it has the original class ids and the inferred does not
    for id, anns in origin_image_id_2_anns.items():
        for ann in anns:
            fname_2_class[origin_id_2_file[ann["image_id"]]] = ann["category_id"]

    # now find the annotations in the inferred that map to the same filename
    for id, anns in infer_image_id_2_anns.items():
        for ann in anns:
            try:
                ann["category_id"] = fname_2_class[infer_id_2_file[id]]
                origin_img_id = origin_file_2_id(infer_file_2_id[id])
                ann["image_id"] = origin_img_id
            except KeyError as e:
                if show_errors:
                    print("The key {} of the inferred json appears to not have any annotations".format(e))

    # now take all the categories from the original # don't add the original images back..dunno why but it throws it off.
    cvj_infer["categories"] = cvj_origin["categories"]


    return cvj_infer

def find_extensions(cvj_obj):
    for img in cvj_obj["images"]:
        file_name, extension = os.path.splitext(img["file_name"])
        for img in os.listdir(cvj_obj.image_folder_path):
            name, ext = os.path.splitext(img)
            if extension != ext:
                for img in cvj_obj["images"]:
                    img["file_name"] = img["file_name"] + ext

                break
        break

    return cvj_obj



