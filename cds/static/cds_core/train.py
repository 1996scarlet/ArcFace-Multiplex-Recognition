# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

import align.facenet as facenet
import os
from os.path import join

import argparse
import sys
import time
import pickle

from sklearn import metrics


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import matplotlib.font_manager
from sklearn import svm

from sklearn.naive_bayes import BernoulliNB
import cv2


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('load_type', type=str,
                        help='Tpye to process.')
    # parser.add_argument('--danger_threshold', type=float,
    #                     help='Danger threshold for judgement.', default=0.985)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.85)
    return parser.parse_args(argv)


def load_data(image_paths, image_size):
    nrof_samples = len(image_paths)
    images = []
    for i in range(nrof_samples):
        img = cv2.imread(image_paths[i])

        aligned = cv2.resize(img, (image_size, image_size),
                             interpolation=cv2.INTER_CUBIC)
        images.append(aligned)
    return images


args = parse_arguments(sys.argv[1:])
load_type = args.load_type
# danger_threshold = args.danger_threshold
image_size = 160

dir_positive = './embedded/positive.npy'
dir_negative = './embedded/negative.npy'
dir_safe = './embedded/safe.npy'
dir_train = './embedded/train.npy'

data_positive = './Temp/true_data'
data_negative = './Temp/false_data'
data_safe = './Temp/safe_data'
data_train = './Temp/train_data'

dataset_positive = facenet.get_dataset(data_positive)
dataset_negative = facenet.get_dataset(data_negative)
dataset_safe = facenet.get_dataset(data_safe)
dataset_train = facenet.get_dataset(data_train)


def generate_emb_array():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    np.random.seed(seed=233)

    import tensorflow as tf
    with tf.Graph().as_default():

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))

        with tf.Session() as sess:
            # =====================PRE LOAD MODEL=========================

            facenet.load_model('model/20180402-114759')

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            '''
            # =====================SAFE DATA=======================

            paths, labels = facenet.get_image_paths_and_labels(dataset_safe)

            safe_images = load_data(sorted(paths), image_size)

            feed_dict = {images_placeholder: safe_images,
                         phase_train_placeholder: False}

            safe_emb_array = sess.run(embeddings, feed_dict=feed_dict)

            np.save(dir_safe, safe_emb_array)

            # ====================TRUE DATA====================

            paths, labels = facenet.get_image_paths_and_labels(
                dataset_positive)

            positive_images = load_data(paths, image_size)

            positive_emb_array = []

            for i in range((len(positive_images) >> 10)+1):

                feed_dict = {images_placeholder: positive_images[i << 10: (i << 10)+1024],
                             phase_train_placeholder: False}

                positive_emb_array.append(
                    sess.run(embeddings, feed_dict=feed_dict))

            positive_emb_array = np.vstack(
                positive_emb_array).reshape(-1, 512)

            np.save(dir_positive, positive_emb_array)

            # ====================FALSE DATA====================

            paths, labels = facenet.get_image_paths_and_labels(
                dataset_negative)

            negative_images = load_data(paths, image_size)

            negative_emb_array = []

            for i in range((len(negative_images) >> 10)+1):

                feed_dict = {images_placeholder: negative_images[i << 10: (i << 10)+1024],
                             phase_train_placeholder: False}

                negative_emb_array.append(
                    sess.run(embeddings, feed_dict=feed_dict))

            negative_emb_array = np.vstack(negative_emb_array).reshape(-1, 512)

            np.save(dir_negative, negative_emb_array)

            '''
            # ====================TRAIN DATA====================

            paths, labels = facenet.get_image_paths_and_labels(
                dataset_train)

            train_images = load_data(paths, image_size)

            train_emb_array = []

            for i in range((len(train_images) >> 10)+1):

                feed_dict = {images_placeholder: train_images[i << 10: (i << 10)+1024],
                             phase_train_placeholder: False}

                train_emb_array.append(
                    sess.run(embeddings, feed_dict=feed_dict))

            train_emb_array = np.vstack(train_emb_array).reshape(-1, 512)

            np.save(dir_train, train_emb_array)

    return safe_emb_array, positive_emb_array, negative_emb_array, train_emb_array


try:
    positive_emb_array = np.load(dir_positive)
    negative_emb_array = np.load(dir_negative)
    safe_emb_array = np.load(dir_safe)
    train_emb_array = np.load(dir_train)
except OSError:
    safe_emb_array, positive_emb_array, negative_emb_array, train_emb_array = generate_emb_array()

print("positive:", positive_emb_array.shape)
print("negative:", negative_emb_array.shape)
print("safe:", safe_emb_array.shape)
print("train:", train_emb_array.shape)

distance_type = ['E', 'C', 'M']


def deal_with_threshold(th, model, negative_array, positive_array):
    negative_predict = model.predict_proba(negative_array)
    positive_predict = model.predict_proba(positive_array)

    negative_after_deal = np.reshape(
        [negative_array[index] for index, item in enumerate(negative_predict)
         if item.max() > th], (-1, 512))

    positive_after_deal = np.reshape(
        [positive_array[index] for index, item in enumerate(positive_predict)
         if item.max() > th], (-1, 512))

    return negative_after_deal, positive_after_deal


if load_type == 'hist':

    plt.figure(1)

    predict_res = model.predict_proba(negative_emb_array)

    for i in range(5):

        plt.subplot(2, 5, i+1)

        danger_threshold = (999 - 20*i)/1000

        dist = [facenet.distance(
            negative_emb_array[index].reshape(1, 512),
            safe_emb_array[item].reshape(1, 512), 1)
            for index, item in enumerate(predict_res.argmax(1))
            if predict_res[index, item] > danger_threshold]

        dist = np.hstack(dist)
        # dist = (dist-min(dist))/(max(dist)-min(dist))

        plt.hist(dist, bins=100)
        plt.title("{:.0f}% {} with {:.2f} thd".format(
            len(dist) / len(predict_res) * 100, "negative", danger_threshold))

    predict_res = model.predict_proba(positive_emb_array)

    for i in range(5):

        plt.subplot(2, 5, i+6)

        danger_threshold = (999 - 20*i)/1000

        dist = [facenet.distance(
            positive_emb_array[index].reshape(1, 512),
            safe_emb_array[item].reshape(1, 512), 1)
            for index, item in enumerate(predict_res.argmax(1))
            if predict_res[index, item] > danger_threshold]

        dist = np.hstack(dist)
        dist = dist[dist > 0.1]
        # dist = (dist-min(dist))/(max(dist)-min(dist))
        plt.hist(dist, bins=100)
        plt.title("{:.0f}% {} with {:.2f} thd".format(
            len(dist) / len(predict_res) * 100, "positive", danger_threshold))

    plt.show()

elif load_type == 'tsne':

    plt.figure(1)

    paths, labels = facenet.get_image_paths_and_labels(
        dataset_positive)

    for i in range(5):

        plt.subplot(1, 5, i+1)

        tsne = TSNE(n_components=2, init='pca',
                    random_state=233, learning_rate=100*i + 10, perplexity=100)
        X_tsne = tsne.fit_transform(positive_emb_array)

        print("Org data dimension is {}. Embedded data dimension is {}".format(
            positive_emb_array.shape[-1], X_tsne.shape[-1]))

        '''嵌入空间可视化'''
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        # plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(labels[i]), color=plt.cm.Set1(labels[i]),
                     fontdict={'weight': 'bold', 'size': 6})
        # plt.xticks([])
        # plt.yticks([])
    plt.show()

elif load_type == 'true':
    plt.figure(1)

    predict_res = model.predict_proba(negative_emb_array)
    dist_list = []

    danger_threshold = 0.985

    for distance_code, distance_name in enumerate(distance_type):

        dist = [facenet.distance(
            negative_emb_array[index, :].reshape(1, 512),
            safe_emb_array[item, :].reshape(1, 512), distance_code)
            for index, item in enumerate(predict_res.argmax(1))
            if predict_res[index, item] > danger_threshold]

        dist = np.hstack(dist)
        if distance_name is 'M':
            dist = dist[dist > 1.34]
        dist = (dist-min(dist))/(max(dist)-min(dist))
        dist_list.append(dist)

        plt.subplot(1, 4, distance_code+1)
        plt.hist(dist, bins=100)
        plt.title("{:.0f}% {} with distance type {}".format(
            len(dist) / len(predict_res) * 100, "positives", distance_name))

    plt.subplot(1, 4, 4)
    plt.hist(dist_list[0]+dist_list[1], bins=100)
    plt.title("{:.0f}% {} with distance type {}".format(
        len(dist) / len(predict_res) * 100, "positives", "E+C"))

    plt.show()

elif load_type == 'append':
    plt.figure(1)
    danger_threshold = 0

    negative_predict_res = model.predict_proba(negative_emb_array)
    positive_predict_res = model.predict_proba(positive_emb_array)

    # print (positive_predict_res)
    # print (positive_predict_res.argmax(1))

    for i, distance_name in enumerate(distance_type):

        plt.subplot(len(distance_type), 1, i+1)

        dist_n = np.hstack([facenet.distance(
            negative_emb_array[index].reshape(1, 512),
            safe_emb_array[item].reshape(1, 512), i)
            for index, item in enumerate(negative_predict_res.argmax(1))
            if negative_predict_res[index, item] > danger_threshold])

        dist_p = np.hstack([facenet.distance(
            positive_emb_array[index].reshape(1, 512),
            safe_emb_array[item].reshape(1, 512), i)
            for index, item in enumerate(positive_predict_res.argmax(1))
            if positive_predict_res[index, item] > danger_threshold])

        if distance_name is 'M':
            dist_n = dist_n[dist_n > 1.34]
            dist_p = dist_p[dist_p > 1.34]

        # dist_n = (dist_n-min(dist_n))/(max(dist_n)-min(dist_n))
        # dist_p = (dist_p-min(dist_p))/(max(dist_p)-min(dist_p))

        plt.hist(dist_n, bins=100, facecolor='yellowgreen',
                 alpha=0.5, label='negative')
        plt.hist(dist_p, bins=100, facecolor='blue',
                 alpha=0.5, label='positive')
        plt.title("{:.2f} thd positive vs negative in {} distance".format(
            danger_threshold, distance_name))
        plt.legend()

    plt.show()

elif load_type == 'svm':

    # ===== 0.998 =====
    # positive: 0.26884779516358465
    # negative: 0.009317183548515906

    threshold = [.9999, .9995, .9996, .9997]

    paths, labels = facenet.get_image_paths_and_labels(
        dataset_train)

    clf = svm.NuSVC(cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
                    max_iter=-1, nu=0.1, probability=True, random_state=233,
                    shrinking=True, tol=0.01, verbose=False)

    clf.fit(train_emb_array, labels)

    _, positive_labels = facenet.get_image_paths_and_labels(
        dataset_positive)

    _, negative_labels = facenet.get_image_paths_and_labels(
        dataset_negative)

    negative_prob = clf.predict_proba(negative_emb_array).max(1)
    positive_prob = clf.predict_proba(positive_emb_array).max(1)

    # plt.figure(1)
    # plt.subplot(2, 1, 1)
    # plt.hist(negative_prob, bins=200)

    # plt.subplot(2, 1, 2)
    # plt.hist(positive_prob, bins=200)
    # plt.show()

    for th in threshold:
        print('='*5, th, '='*5)
        print('positive:',
              positive_prob[positive_prob > th].size/positive_prob.size)
        print('negative:',
              negative_prob[negative_prob > th].size/negative_prob.size)

    classifier_filename_exp = os.path.expanduser('./model/svm.pkl')

    class_names = [cls.name.replace('_', ' ') for cls in dataset_train]
    print(class_names)

    with open(classifier_filename_exp, 'wb') as outfile:
        pickle.dump((clf, class_names), outfile)

    print('positive:', clf.score(positive_emb_array, positive_labels))

    # print('negative:', negative_prob)
    # print('positive:', positive_prob)

    # y_pred_train = clf.predict(train_emb_array)
    # y_pred_test = clf.predict(positive_emb_array)
    # y_pred_outliers = clf.predict(negative_emb_array)

    # n_error_train = y_pred_train[y_pred_train == -1].size
    # n_error_test = y_pred_test[y_pred_test == -1].size
    # n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    # print("train error:{}/{}".format(n_error_train, y_pred_train.size))
    # print("test error:{}/{}".format(n_error_test, y_pred_test.size))
    # print("outliers error:{}/{}".format(n_error_outliers, y_pred_outliers.size))
elif load_type == 'test':
    threshold = [.9, .9999, .999]

    classifier_layer_one = os.path.expanduser('./model/layer_one.pkl')
    classifier_layer_two = os.path.expanduser('./model/strick.pkl')
    classifier_layer_three = os.path.expanduser('./model/svm.pkl')
    classifier_layer_four = os.path.expanduser('./model/layer_four.pkl')
    classifier_layer_five = os.path.expanduser('./model/layer_five.pkl')
    classifier_layer_six = os.path.expanduser('./model/layer_six.pkl')
    classifier_layer_seven = os.path.expanduser('./model/layer_seven.pkl')
    classifier_layer_eight = os.path.expanduser('./model/layer_eight.pkl')

    with open(classifier_layer_one, 'rb') as infile:
        (layer_one_model, _) = pickle.load(infile)

    with open(classifier_layer_two, 'rb') as infile:
        (layer_two_model, _) = pickle.load(infile)

    with open(classifier_layer_three, 'rb') as infile:
        (layer_three_model, _) = pickle.load(infile)

    with open(classifier_layer_four, 'rb') as infile:
        (layer_four_model, _) = pickle.load(infile)

    with open(classifier_layer_five, 'rb') as infile:
        (layer_five_model, _) = pickle.load(infile)

    with open(classifier_layer_six, 'rb') as infile:
        (layer_six_model, _) = pickle.load(infile)

    with open(classifier_layer_seven, 'rb') as infile:
        (layer_seven_model, _) = pickle.load(infile)

    with open(classifier_layer_eight, 'rb') as infile:
        (layer_eight_model, _) = pickle.load(infile)

    negative_prob = layer_three_model.predict_proba(negative_emb_array).max(1)
    positive_prob = layer_three_model.predict_proba(positive_emb_array).max(1)

    for th in threshold:
        print('='*5, th, '='*5)
        print('positive:',
              positive_prob[positive_prob > th].size/positive_prob.size)
        print('negative:',
              negative_prob[negative_prob > th].size/negative_prob.size)

elif load_type == 'train':

    paths, labels = facenet.get_image_paths_and_labels(dataset_train)

    X_train, X_test, y_train, y_test = train_test_split(
        train_emb_array, labels, test_size=.01, random_state=233)

    # hidden_layer_sizes=(512, 256, 128, 256, 512),
    model = MLPClassifier(hidden_layer_sizes=(2048),
                          activation='relu', solver='adam', random_state=233,
                          learning_rate_init=0.0001, max_iter=20000)

    model.fit(X_train, y_train)

    class_names = [cls.name.replace('_', ' ') for cls in dataset_train]
    print(class_names)

    print('======MODEL SAVED======')
    print('Number of classes: %d' % len(dataset_train))
    print('Number of images: %d' % len(paths))
    print('============================================>>>>')

    # with open(classifier_filename_exp, 'rb') as infile:
    #     (model, class_names) = pickle.load(infile)

    _, train_test = facenet.get_image_paths_and_labels(dataset_train)

    print('train dataset accuracy: %.2f%%' %
          (100 * metrics.accuracy_score(train_test, model.predict(train_emb_array))))

    _, positive_test = facenet.get_image_paths_and_labels(dataset_positive)

    text_acc = 100 * \
        metrics.accuracy_score(
            positive_test, model.predict(positive_emb_array))
    print('positive dataset accuracy: %.2f%%' % (text_acc))

    # classifier_filename_exp = os.path.expanduser('{}.pkl'.format(text_acc))
    classifier_filename_exp = os.path.expanduser('./model/mlp.pkl')

    with open(classifier_filename_exp, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)

    negative_prob = model.predict_proba(negative_emb_array).max(1)
    positive_prob = model.predict_proba(positive_emb_array).max(1)

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.hist(negative_prob, bins=200)

    plt.subplot(2, 1, 2)
    plt.hist(positive_prob, bins=200)
    plt.show()

elif load_type == 'mlp':

    threshold = [1-10e-7, 1-10e-6, 1-10e-5, 1-10e-4, 1-10e-3,
                 1-10e-2, 1-10e-1, 1-10e-10]

    classifier_layer_one = os.path.expanduser('./model/layer_one.pkl')
    classifier_layer_two = os.path.expanduser('./model/layer_two.pkl')
    classifier_layer_three = os.path.expanduser('./model/layer_three.pkl')
    classifier_layer_four = os.path.expanduser('./model/layer_four.pkl')
    classifier_layer_five = os.path.expanduser('./model/layer_five.pkl')
    classifier_layer_six = os.path.expanduser('./model/layer_six.pkl')
    classifier_layer_seven = os.path.expanduser('./model/layer_seven.pkl')
    classifier_layer_eight = os.path.expanduser('./model/layer_eight.pkl')

    with open(classifier_layer_one, 'rb') as infile:
        (layer_one_model, _) = pickle.load(infile)

    with open(classifier_layer_two, 'rb') as infile:
        (layer_two_model, _) = pickle.load(infile)

    with open(classifier_layer_three, 'rb') as infile:
        (layer_three_model, _) = pickle.load(infile)

    with open(classifier_layer_four, 'rb') as infile:
        (layer_four_model, _) = pickle.load(infile)

    with open(classifier_layer_five, 'rb') as infile:
        (layer_five_model, _) = pickle.load(infile)

    with open(classifier_layer_six, 'rb') as infile:
        (layer_six_model, _) = pickle.load(infile)

    with open(classifier_layer_seven, 'rb') as infile:
        (layer_seven_model, _) = pickle.load(infile)

    with open(classifier_layer_eight, 'rb') as infile:
        (layer_eight_model, _) = pickle.load(infile)

    print('='*18)

    # ======================LAYER ONE============================

    negative_predict = layer_one_model.predict_proba(negative_emb_array)
    positive_predict = layer_one_model.predict_proba(positive_emb_array)

    negative_after_one = np.reshape(
        [negative_emb_array[index] for index, item in enumerate(negative_predict)
         if item.max() <= threshold[0]], (-1, 512))

    positive_after_one = np.reshape(
        [positive_emb_array[index] for index, item in enumerate(positive_predict)
         if item.max() <= threshold[0]], (-1, 512))

    print("Layer one TH0:", threshold[0])
    print("negative:",
          negative_after_one.shape[0]/negative_emb_array.shape[0])
    print("positive:",
          positive_after_one.shape[0]/positive_emb_array.shape[0])
    print('='*18)

    # ======================LAYER TWO============================

    negative_predict = layer_two_model.predict_proba(negative_after_one)
    positive_predict = layer_two_model.predict_proba(positive_after_one)

    negative_after_two = np.reshape(
        [negative_after_one[index] for index, item in enumerate(negative_predict)
         if item.max() <= threshold[1]], (-1, 512))

    positive_after_two = np.reshape(
        [positive_after_one[index] for index, item in enumerate(positive_predict)
         if item.max() <= threshold[1]], (-1, 512))

    print("Layer two TH1:", threshold[1])
    print("negative:",
          negative_after_two.shape[0]/negative_emb_array.shape[0])
    print("positive:",
          positive_after_two.shape[0]/positive_emb_array.shape[0])
    print('='*18)

    # ======================LAYER THREE============================

    negative_predict = layer_three_model.predict_proba(negative_after_two)
    positive_predict = layer_three_model.predict_proba(positive_after_two)

    negative_after_three = np.reshape(
        [negative_after_two[index] for index, item in enumerate(negative_predict)
         if item.max() <= threshold[2]], (-1, 512))

    positive_after_three = np.reshape(
        [positive_after_two[index] for index, item in enumerate(positive_predict)
         if item.max() <= threshold[2]], (-1, 512))

    print("Layer three TH2:", threshold[2])
    print("negative:",
          negative_after_three.shape[0]/negative_emb_array.shape[0])
    print("positive:",
          positive_after_three.shape[0]/positive_emb_array.shape[0])
    print('='*18)

    # ======================LAYER FOUR============================

    negative_predict = layer_four_model.predict_proba(negative_after_three)
    positive_predict = layer_four_model.predict_proba(positive_after_three)

    negative_after_four = np.reshape(
        [negative_after_three[index] for index, item in enumerate(negative_predict)
         if item.max() <= threshold[3]], (-1, 512))

    positive_after_four = np.reshape(
        [positive_after_three[index] for index, item in enumerate(positive_predict)
         if item.max() <= threshold[3]], (-1, 512))

    print("Layer four TH3:", threshold[3])
    print("negative:",
          negative_after_four.shape[0]/negative_emb_array.shape[0])
    print("positive:",
          positive_after_four.shape[0]/positive_emb_array.shape[0])
    print('='*18)

    # ======================LAYER FIVE============================

    negative_predict = layer_five_model.predict_proba(negative_after_four)
    positive_predict = layer_five_model.predict_proba(positive_after_four)

    negative_after_five = np.reshape(
        [negative_after_four[index] for index, item in enumerate(negative_predict)
         if item.max() <= threshold[4]], (-1, 512))

    positive_after_five = np.reshape(
        [positive_after_four[index] for index, item in enumerate(positive_predict)
         if item.max() <= threshold[4]], (-1, 512))

    print("Layer five TH4:", threshold[4])
    print("negative:",
          negative_after_five.shape[0]/negative_emb_array.shape[0])
    print("positive:",
          positive_after_five.shape[0]/positive_emb_array.shape[0])
    print('='*18)

    # ======================TP->50%;FR->4%============================
    # ======================LAYER SIX============================

    negative_predict = layer_six_model.predict_proba(negative_after_five)
    positive_predict = layer_six_model.predict_proba(positive_after_five)

    negative_after_six = np.reshape(
        [negative_after_five[index] for index, item in enumerate(negative_predict)
         if item.max() <= threshold[5]], (-1, 512))

    positive_after_six = np.reshape(
        [positive_after_five[index] for index, item in enumerate(positive_predict)
         if item.max() <= threshold[5]], (-1, 512))

    print("Layer six TH5:", threshold[5])
    print("negative:",
          negative_after_six.shape[0]/negative_emb_array.shape[0])
    print("positive:",
          positive_after_six.shape[0]/positive_emb_array.shape[0])
    print('='*18)

    # ======================LAYER SEVEN============================

    negative_predict = layer_seven_model.predict_proba(negative_after_six)
    positive_predict = layer_seven_model.predict_proba(positive_after_six)

    negative_after_seven = np.reshape(
        [negative_after_six[index] for index, item in enumerate(negative_predict)
         if item.max() <= threshold[6]], (-1, 512))

    positive_after_seven = np.reshape(
        [positive_after_six[index] for index, item in enumerate(positive_predict)
         if item.max() <= threshold[6]], (-1, 512))

    print("Layer seven TH6:", threshold[6])
    print("negative:",
          negative_after_seven.shape[0]/negative_emb_array.shape[0])
    print("positive:",
          positive_after_seven.shape[0]/positive_emb_array.shape[0])
    print('='*18)

    # ======================LAYER EIGHT============================

    negative_predict = layer_eight_model.predict_proba(negative_after_seven)
    positive_predict = layer_eight_model.predict_proba(positive_after_seven)

    negative_after_eight = np.reshape(
        [negative_after_seven[index] for index, item in enumerate(negative_predict)
         if item.max() <= threshold[7]], (-1, 512))

    positive_after_eight = np.reshape(
        [positive_after_seven[index] for index, item in enumerate(positive_predict)
         if item.max() <= threshold[7]], (-1, 512))

    print("Layer eight TH7:", threshold[7])
    print("negative:",
          negative_after_eight.shape[0]/negative_emb_array.shape[0])
    print("positive:",
          positive_after_eight.shape[0]/positive_emb_array.shape[0])
    print('='*18)
    # print(
    #     "FR：", negative_after_layer_three.shape[0]/negative_emb_array.shape[0])

    # print(
    #     "TP：", positive_after_layer_three.shape[0]/positive_emb_array.shape[0])

elif load_type == 'inter-simple':

    threshold = [0.8, 0.96, 0.9999]

    classifier_layer_one = os.path.expanduser('./model/layer_one.pkl')
    classifier_layer_two = os.path.expanduser('./model/layer_two.pkl')
    classifier_layer_three = os.path.expanduser('./model/layer_three.pkl')

    with open(classifier_layer_one, 'rb') as infile:
        (layer_one_model, _) = pickle.load(infile)

    with open(classifier_layer_two, 'rb') as infile:
        (layer_two_model, _) = pickle.load(infile)

    with open(classifier_layer_three, 'rb') as infile:
        (layer_three_model, _) = pickle.load(infile)

    print('='*18)

    # ======================LAYER ONE============================

    negative_after_layer_one, positive_after_layer_one = deal_with_threshold(
        threshold[0], layer_one_model, negative_emb_array, positive_emb_array)

    print("Layer one TH0:", threshold[0])
    print("negative:",
          negative_after_layer_one.shape[0]/negative_emb_array.shape[0])
    print("positive:",
          positive_after_layer_one.shape[0]/positive_emb_array.shape[0])
    print('='*18)

    # ======================LAYER TWO============================

    negative_after_layer_two, positive_after_layer_two = deal_with_threshold(
        threshold[1], layer_two_model, negative_emb_array, positive_emb_array)

    print("Layer two TH1:", threshold[1])
    print("negative:",
          negative_after_layer_two.shape[0]/negative_emb_array.shape[0])
    print("positive:",
          positive_after_layer_two.shape[0]/positive_emb_array.shape[0])
    print('='*18)

    # ======================LAYER TWO============================

    negative_after_layer_three, positive_after_layer_three = deal_with_threshold(
        threshold[2], layer_three_model, negative_emb_array, positive_emb_array)

    print("Layer three TH2:", threshold[2])
    print("negative:",
          negative_after_layer_three.shape[0]/negative_emb_array.shape[0])
    print("positive:",
          positive_after_layer_three.shape[0]/positive_emb_array.shape[0])
    print('='*18)

    # ======================INTER 1 VS 2============================

    positive_with_one_and_two = np.reshape(
        [item for item in positive_after_layer_one
         if (item in positive_after_layer_two and item in positive_after_layer_three)], (-1, 512)
    )

    negative_with_one_and_two = np.reshape(
        [item for item in negative_after_layer_one
         if (item in negative_after_layer_two and item in negative_after_layer_three)], (-1, 512)
    )

    print('FR:', negative_with_one_and_two.shape[0],
          negative_with_one_and_two.shape[0]/negative_emb_array.shape[0])

    print('TP:', positive_with_one_and_two.shape[0],
          positive_with_one_and_two.shape[0]/positive_emb_array.shape[0])

elif load_type == 'distance':

    threshold = [0.75, 0.65, 0.6]

    classifier_layer_one = os.path.expanduser('./model/layer_one.pkl')
    classifier_layer_two = os.path.expanduser('./model/layer_two.pkl')
    classifier_layer_three = os.path.expanduser('./model/layer_three.pkl')

    with open(classifier_layer_one, 'rb') as infile:
        (layer_one_model, _) = pickle.load(infile)

    with open(classifier_layer_two, 'rb') as infile:
        (layer_two_model, _) = pickle.load(infile)

    with open(classifier_layer_three, 'rb') as infile:
        (layer_three_model, _) = pickle.load(infile)

    print('='*18)

    # ======================LAYER ONE============================

    negative_after_layer_one, positive_after_layer_one = deal_with_threshold(
        threshold[0], layer_one_model, negative_emb_array, positive_emb_array)

    print("Layer one TH0:", threshold[0])
    print("negative:",
          negative_after_layer_one.shape[0]/negative_emb_array.shape[0])
    print("positive:",
          positive_after_layer_one.shape[0]/positive_emb_array.shape[0])
    print('='*18)

    # ======================LAYER TWO============================

    negative_after_layer_two, positive_after_layer_two = deal_with_threshold(
        threshold[1], layer_two_model, negative_emb_array, positive_emb_array)

    print("Layer two TH1:", threshold[1])
    print("negative:",
          negative_after_layer_two.shape[0]/negative_emb_array.shape[0])
    print("positive:",
          positive_after_layer_two.shape[0]/positive_emb_array.shape[0])
    print('='*18)

    # ======================LAYER TWO============================

    negative_after_layer_three, positive_after_layer_three = deal_with_threshold(
        threshold[2], layer_three_model, negative_emb_array, positive_emb_array)

    print("Layer three TH2:", threshold[2])
    print("negative:",
          negative_after_layer_three.shape[0]/negative_emb_array.shape[0])
    print("positive:",
          positive_after_layer_three.shape[0]/positive_emb_array.shape[0])
    print('='*18)

    negative_predict_res = layer_three_model.predict_proba(
        negative_after_layer_three)

    dist_n = np.hstack([facenet.distance(
        negative_emb_array[index].reshape(1, 512),
        safe_emb_array[item].reshape(1, 512), 0)
        for index, item in enumerate(negative_predict_res.argmax(1))
    ])

    positive_predict_res = layer_three_model.predict_proba(
        positive_after_layer_three)

    dist_p = np.hstack([facenet.distance(
        positive_emb_array[index].reshape(1, 512),
        safe_emb_array[item].reshape(1, 512), 0)
        for index, item in enumerate(positive_predict_res.argmax(1))
    ])

    print('FR:', dist_n[dist_n < 0.9].size/dist_n.size)
    print('TP:', dist_p[dist_p < 0.9].size/dist_p.size)

elif load_type == 'inter-class':

    threshold = [0.98, 0.1, 0.9999]

    classifier_layer_one = os.path.expanduser('./model/layer_one.pkl')
    classifier_layer_two = os.path.expanduser('./model/layer_two.pkl')
    classifier_layer_three = os.path.expanduser('./model/layer_three.pkl')
    classifier_layer_four = os.path.expanduser('./model/layer_four.pkl')

    with open(classifier_layer_one, 'rb') as infile:
        (layer_one_model, _) = pickle.load(infile)

    with open(classifier_layer_two, 'rb') as infile:
        (layer_two_model, _) = pickle.load(infile)

    with open(classifier_layer_three, 'rb') as infile:
        (layer_three_model, _) = pickle.load(infile)

    with open(classifier_layer_four, 'rb') as infile:
        (layer_four_model, _) = pickle.load(infile)

    print('='*18)

    negative_res_one = layer_one_model.predict(negative_emb_array)
    positive_res_one = layer_one_model.predict(positive_emb_array)

    negative_res_two = layer_two_model.predict(negative_emb_array)
    positive_res_two = layer_two_model.predict(positive_emb_array)

    negative_res_three = layer_three_model.predict(negative_emb_array)
    positive_res_three = layer_three_model.predict(positive_emb_array)

    negative_res_four = layer_four_model.predict(negative_emb_array)
    positive_res_four = layer_four_model.predict(positive_emb_array)

    inter_of_negative = np.reshape(
        [item for index, item in enumerate(negative_emb_array)
         if negative_res_one[index] == negative_res_two[index] == negative_res_three[index] == negative_res_four[index]
         ], (-1, 512))

    inter_of_positive = np.reshape(
        [item for index, item in enumerate(positive_emb_array)
         if positive_res_one[index] == positive_res_two[index] == positive_res_three[index] == positive_res_four[index]
         ], (-1, 512))

    negative_after_layer_three, positive_after_layer_three = deal_with_threshold(
        threshold[1], layer_three_model, inter_of_negative, inter_of_positive)

    print(
        'FR:', negative_after_layer_three.shape[0]/negative_emb_array.shape[0])
    print(
        'TP:', positive_after_layer_three.shape[0]/positive_emb_array.shape[0])

elif load_type == 'auto-encoder':
    paths, labels = facenet.get_image_paths_and_labels(
        dataset_train)

    X_train, X_test, y_train, y_test = train_test_split(
        train_emb_array, labels, test_size=.01, random_state=233)

    model = MLPClassifier(hidden_layer_sizes=(1024, 512, 64, 32),
                          activation='relu', solver='adam', random_state=233,
                          learning_rate_init=0.0001, max_iter=20000)

    model.fit(X_train, y_train)

    class_names = [cls.name.replace('_', ' ') for cls in dataset_train]
    print(class_names)

    print('======MODEL SAVED======')
    print('Number of classes: %d' % len(dataset_train))
    print('Number of images: %d' % len(paths))
    print('============================================>>>>')

    # with open(classifier_filename_exp, 'rb') as infile:
    #     (model, class_names) = pickle.load(infile)

    _, train_test = facenet.get_image_paths_and_labels(
        dataset_train)
    print('train dataset accuracy: %.2f%%' %
          (100 * metrics.accuracy_score(train_test, model.predict(train_emb_array))))

    _, positive_test = facenet.get_image_paths_and_labels(
        dataset_positive)

    text_acc = 100 * \
        metrics.accuracy_score(
            positive_test, model.predict(positive_emb_array))
    print('positive dataset accuracy: %.2f%%' % (text_acc))

    classifier_filename_exp = os.path.expanduser('{}.pkl'.format(text_acc))

    with open(classifier_filename_exp, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)

    negative_prob = model.predict_proba(negative_emb_array).max(1)
    positive_prob = model.predict_proba(positive_emb_array).max(1)

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.hist(negative_prob, bins=200)

    plt.subplot(2, 1, 2)
    plt.hist(positive_prob, bins=200)
    plt.show()
