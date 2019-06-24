import sys
import os
import numpy as np
import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn import metrics

from helper import read_pkl_model, start_up_init, get_dataset, get_image_paths_and_labels
import face_embedding
import face_detector

# =================== ARGS ====================
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
args = start_up_init()
args.retina_model = './model/M25'
args.scales = [0.5]

# =================== MODEL CLASS ====================
detector = face_detector.DetectorModel(args)
arcface = face_embedding.EmbeddingModel(args)

# =================== LOAD DATASET ====================.
dir_train = './Temp/train.npy'
data_train = './Temp/train_data'
dataset_train = get_dataset(data_train)
paths_train, labels_train = get_image_paths_and_labels(dataset_train)

try:
    train_emb_array = np.load(dir_train)
except OSError:
    if not os.path.exists('./Temp/raw/'):
        os.makedirs('./Temp/raw/')
    detector.get_all_boxes_from_path(paths_train, save_img=True)
    dataset_train = get_dataset(data_train)
    paths_train, labels_train = get_image_paths_and_labels(dataset_train)
    train_emb_array = arcface.get_features_from_path(paths_train)
    np.save(dir_train, train_emb_array)

print('Train dataset reloaded: ', len(train_emb_array))

class_names = [cls.name.replace('_', ' ') for cls in dataset_train]
print(class_names)

mlp = MLPClassifier(hidden_layer_sizes=(550, ), verbose=True,
                    activation='relu', solver='adam', tol=10e-7, n_iter_no_change=100,
                    learning_rate_init=1e-3, max_iter=50000)

mlp.fit(train_emb_array, labels_train)

with open('./model/mlp.pkl', 'wb') as outfile:
    pickle.dump((mlp, class_names), outfile)
