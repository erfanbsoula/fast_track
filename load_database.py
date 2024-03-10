import os
import cv2

def get_known_ids(feature_extractor, dir='application/tmp'):
    
    ids = []
    features = []
    names = []
    
    for fname in os.listdir(dir):

        if fname[0] == '.':
            continue

        known_id, name = fname.split('-')
        name = name.split('.')[0]
        img = cv2.imread(os.path.join(dir, fname))
        feature = feature_extractor([img])[0]
        features.append(feature)
        ids.append(int(known_id))
        names.append(name)

    return ids, features, names