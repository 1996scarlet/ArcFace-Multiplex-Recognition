import face_embedding
import dataset_gen as dg
import argparse
import cv2
import sys
import numpy as np
import time
from termcolor import colored
import matplotlib.pyplot as plt
import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn import metrics

parser = argparse.ArgumentParser(description='arcface recognize model train')
# general
parser.add_argument('--image-size', default='112,112',
                    help='')
parser.add_argument('--model', default='./model-r100-ii/model,0',
                    help='path to load model.')
parser.add_argument('--gpu', default=-1, type=int,
                    help='gpu id, -1 for cpu')
parser.add_argument('--det', default=1, type=int,
                    help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=1, type=int,
                    help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float,
                    help='ver dist threshold')

# =================== ARGS ====================
args = parser.parse_args()

# =================== MODEL CLASS ====================
model = face_embedding.FaceModel(args)

# =================== LOAD DATASET ====================.
dir_train = './Embedding/train.npy'
data_train = './Temp/train_data'
dataset_train = dg.get_dataset(data_train)
paths_train, labels_train = dg.get_image_paths_and_labels(dataset_train)

dir_safe = './Embedding/safe.npy'
data_safe = './Temp/safe_data'
dataset_safe = dg.get_dataset(data_safe)
paths_safe, labels_safe = dg.get_image_paths_and_labels(dataset_safe)

dir_danger = './Embedding/danger.npy'
data_danger = './Temp/danger_data'
dataset_danger = dg.get_dataset(data_danger)
paths_danger, labels_danger = dg.get_image_paths_and_labels(dataset_danger)

# train_emb_array = model.get_all_feature(paths)

# # for index, img in enumerate(paths):
# #     print(index, 450)
# # train_emb_array = [model.get_feature(cv2.imread(img)) and
# #                    print(index, len(paths)) for index, img in enumerate(paths)]

# np.save(dir_train, train_emb_array)
# print(len(train_emb_array))
try:
    train_emb_array = np.load(dir_train)
except OSError:
    train_emb_array = model.get_all_feature(paths_train)
    np.save(dir_train, train_emb_array)

print('Train dataset reloaded: ', len(train_emb_array))

try:
    safe_emb_array = np.load(dir_safe)
except OSError:
    safe_emb_array = model.get_all_feature(paths_safe)
    np.save(dir_safe, safe_emb_array)

print('Safe dataset reloaded: ', len(safe_emb_array))

try:
    danger_emb_array = np.load(dir_danger)
except OSError:
    danger_emb_array = model.get_all_feature(paths_danger)
    np.save(dir_danger, danger_emb_array)

print('Danger dataset reloaded: ', len(danger_emb_array))

# print(labels)

class_names = [cls.name.replace('_', ' ') for cls in dataset_train]
print(class_names)

# X_train, X_test, y_train, y_test = train_test_split(train_emb_array, labels)

mlp = MLPClassifier(hidden_layer_sizes=(550, ), verbose=True,
                    activation='relu', solver='adam', tol=10e-7, n_iter_no_change=100,
                    learning_rate_init=10e-4, max_iter=50000)
# model.fit(X_train, y_train)
# # model.fit(train_emb_array, labels)
# print('train dataset accuracy: %.2f%%' %
#       (100 * metrics.accuracy_score(y_test, model.predict(X_test))))

# mlp.fit(train_emb_array, labels_train)

with open('./model-mlp/mlp.pkl', 'rb') as infile:
    (mlp, class_names) = pickle.load(infile)

safe_prob = mlp.predict_proba(safe_emb_array)
danger_prob = mlp.predict_proba(danger_emb_array)


# with open('./model-mlp/mlp.pkl', 'wb') as outfile:
#     pickle.dump((mlp, class_names), outfile)

plt.hist(safe_prob.max(1), bins=10000,
         facecolor='blue', alpha=0.5, label='safe-arcface')
plt.hist(danger_prob.max(1), bins=10000,
         facecolor='red', alpha=0.5, label='danger-arcface')

with open('./model-mlp/mlp-facenet.pkl', 'rb') as infile:
    (mlp, class_names) = pickle.load(infile)

safe_prob = mlp.predict_proba(safe_emb_array)
danger_prob = mlp.predict_proba(danger_emb_array)

plt.hist(safe_prob.max(1), bins=10000,
         facecolor='yellow', alpha=0.5, label='safe-facenet')
plt.hist(danger_prob.max(1), bins=10000,
         facecolor='green', alpha=0.5, label='danger-facenet')

plt.legend()
plt.title("Split into %d parts of the training sample" % len(class_names))
plt.show()
