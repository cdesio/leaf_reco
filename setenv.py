import sys, os

def add_folders(key=None):
    if key=='deepthought':
    	src_folder = os.path.join("/","storage", "yw18581", "src", "leaf_reco")
    elif key=='iron-hide':
        src_folder = os.path.join("~/","HEP","uob","leaf_reco")	
    elif(key=='notebook'):
        src_folder = '../'

    ml_models = os.path.join(src_folder, "ml_models")
    data_prep = os.path.join(src_folder, "data_preparation")
    utilities = os.path.join(src_folder, "utilities")

    sys.path.append(ml_models)
    sys.path.append(utilities)
    sys.path.append(data_prep)
    return

"""
def add_folders():
   # src_folder = os.path.join("/","storage", "yw18581", "src", "leaf_reco")

    ml_models = "ml_models"
    data_prep = "data_preparation"
    utilities = "utilities"

    sys.path.append(ml_models)
    sys.path.append(utilities)
    sys.path.append(data_prep)
    return
"""
