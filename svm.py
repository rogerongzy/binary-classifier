import random
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn import svm


def read_data(filename_data_train, filename_label_train, filename_data_test, suf):
    data_train_dict = scio.loadmat(filename_data_train + suf) # dict
    label_train_dict = scio.loadmat(filename_label_train + suf)
    data_test_dict = scio.loadmat(filename_data_test + suf)

    data_train = np.array(data_train_dict[filename_data_train]) # ndarray
    label_train = np.array(label_train_dict[filename_label_train])
    data_test = np.array(data_test_dict[filename_data_test])

    data_train_ = np.hstack((data_train, label_train)) # 330, 34, full
    return data_train_, data_test


def split_label(data_):
    [data, label] = np.split(data_, [data_.shape[1] - 1], axis=1)
    return data, label


def random_shuffle(data_):
    shape = data_.shape
    data_shuffled_ = np.array([])
    
    for i in range(data_.shape[0]):
        index = random.randint(0, data_.shape[0] - 1)
        data_shuffled_ = np.append(data_shuffled_, data_[index, :])
        data_ = np.delete(data_, index, axis=0)
    
    data_shuffled_ = data_shuffled_.reshape(shape)

    return data_shuffled_


def random_split_validation(data_, ratio):
    if ratio > 1 or ratio <= 0:
        ratio = 0.7
    num_full = data_.shape[0]
    num_validation = int(num_full * ratio)
    
    validation_ = np.array([]) # empty

    # select a certain row randomly
    for i in range(num_full - num_validation):
        index = random.randint(0, data_.shape[0] - 1)
        # print(data_[index, :])
        validation_ = np.append(validation_, data_[index, :])
        data_ = np.delete(data_, index, axis=0)

    validation_ = validation_.reshape(num_full - num_validation, data_.shape[1])

    return data_, validation_


def svm_train(data_train_, c, gm):
    data_train, label_train = split_label(data_train_)
    svm_model = svm.SVC(C=c, gamma=gm, kernel='rbf', probability=True) # C=1.0, gamma='auto'
    svm_model.fit(data_train, label_train.ravel()) # reshape for input
    return svm_model


def svm_validate(svm_model, data_validation_):
    # print(svm_model.score(split_label(data_validation_)[0], split_label(data_validation_)[1]))
    return svm_model.score(split_label(data_validation_)[0], split_label(data_validation_)[1])


def get_score_list(data_, c, gm):
    score_list = []
    for i in range(5000):
        data_train_, data_validation_ = random_split_validation(random_shuffle(data_), 0.7)
        svm_model = svm_train(data_train_, c, gm)
        score = svm_validate(svm_model, data_validation_)
        score_list.append(score)
    return score_list


def print_score_hist(score_list):
    plt.hist(score_list, range=(0.85,1), bins=30)
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.savefig('score_hist_1.0_auto.png')
    # plt.show()


def generate_scores_map(data_):
    for m in range(-5, 15 + 1, 1):
        for n in range(-15, 3 + 1, 1):

            score_list = get_score_list(data_, 2**m, 2**n)

            str1 = 'c = ' + str(2**m)
            str2 = 'gm = ' + str(2**n)
            str3 = 'score = ' + str(max(set(score_list), key = score_list.count))
            print(str1.ljust(10), str2.ljust(20), str3.ljust(30))


def svm_predict(svm_model, data_test):
    print(svm_model.predict(data_test))


def svm_evaluate(svm_model, data_validation_):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    data = split_label(data_validation_)[0]
    label = split_label(data_validation_)[1].ravel()
    pred = svm_model.predict(data)

    for i in range(label.shape[0]):
        if label[i] == 1:
            if pred[i] == 1:
                TP = TP + 1
            elif pred[i] == -1:
                FN = FN + 1
        if label[i] == -1:
            if pred[i] == 1:
                FP = FP + 1
            elif pred[i] == -1:
                TN = TN + 1
    
    print('Accuracy: ' + str((TP+TN)/(TP+FP+FN+TN)))
    print('Precision: ' + str(TP/(TP+FP)))
    print('Recall: ' + str(TP/(TP+FN)))
        

if __name__ == '__main__':
    data_, data_test = read_data('data_train', 'label_train', 'data_test', '.mat')
    
    data_train_, data_validation_ = random_split_validation(random_shuffle(data_), 0.7)
    svm =  svm_train(data_, 1, 0.25)
    svm_predict(svm, data_test)
    
    svm_evaluate(svm, data_validation_)