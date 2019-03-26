from cvjson.cvj import CVJ
from cvjson.extensions.visualizer import Visualizer

train_json_path = "/home/ben/Desktop/completed_train_refinement.json"
train_image_path = "/home/ben/Desktop/zvezda_data/mardct/Completed_mardct_refinement/completed_train_refinement"

test_json_path = "/home/ben/Desktop/completed_test_refinement.json"
test_image_path = "/home/ben/Desktop/zvezda_data/mardct/Completed_mardct_refinement/completed_refinement_test_images"

# train_cvj = CVJ(train_json_path, train_image_path)
test_cvj = CVJ(test_json_path, test_image_path)


Visualizer(test_cvj).view_images_by_name(["1171.jpg"])

