
from sklearn import svm
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import pickle
import os


parser = argparse.ArgumentParser(description='Process args for DAG SVM Classifer')
parser.add_argument("--kirc_train_file", type=str, required=False)
parser.add_argument("--kirp_train_file", type=str, required=False)
parser.add_argument("--kich_train_file", type=str, required=False)

parser.add_argument("--kirc_valid_file", type=str, required=True)
parser.add_argument("--kirp_valid_file", type=str, required=True)
parser.add_argument("--kich_valid_file", type=str, required=True)


def train(kirc_file, kirp_file, kich_file):
    kirc, kirp, kich = [np.load(file) for file in [kirc_file, kirp_file, kich_file]]
    X_kirc_kirp = np.concatenate([kirc, kirp], axis=0)
    X_kirp_kich = np.concatenate([kirp, kich], axis=0)
    X_kirc_kich = np.concatenate([kirc, kich], axis=0)

    Y_kirc_kirp = np.concatenate([np.zeros((kirc.shape[0])), np.ones((kirp.shape[0]))])
    Y_kirp_kich = np.concatenate([np.zeros((kirp.shape[0])), np.ones((kich.shape[0]))])
    Y_kirc_kich = np.concatenate([np.zeros((kirc.shape[0])), np.ones((kich.shape[0]))])

    print("Training KIRC-KIRP SVM", flush=True)
    clf_kirc_kirp = svm.SVC(kernel = 'linear')
    clf_kirc_kirp.fit(X_kirc_kirp, Y_kirc_kirp)
    pickle.dump(clf_kirc_kirp, open("svm_kirc_kirp.pickle", 'wb'))

    print("Training KIRP-KICH SVM", flush=True)
    clf_kirp_kich = svm.SVC(kernel = 'linear')
    clf_kirp_kich.fit(X_kirp_kich, Y_kirp_kich)
    pickle.dump(clf_kirp_kich, open("svm_kirp_kich.pickle", 'wb'))

    print("Training KIRC-KICH SVM", flush=True)
    clf_kirc_kich = svm.SVC(kernel = 'linear')
    clf_kirc_kich.fit(X_kirc_kich, Y_kirc_kich)
    pickle.dump(clf_kirc_kich, open("svm_kirc_kich.pickle", 'wb'))

    return clf_kirc_kirp, clf_kirp_kich, clf_kirc_kich


def test(kirc_file, kirp_file, kich_file):
    clf_kirc_kirp, clf_kirp_kich, clf_kirc_kich = [pickle.load(open(f, 'rb')) for f in \
            ["svm_kirc_kirp.pickle", "svm_kirp_kich.pickle", "svm_kirc_kich.pickle"]]
    kirc, kirp, kich = [np.load(file) for file in [kirc_file, kirp_file, kich_file]]
    X = np.concatenate([kirc, kirp, kich])
    Y = np.concatenate([np.full((kirc.shape[0]), 0.), np.full((kirp.shape[0]), 1.),
                        np.full((kich.shape[0]), 2.)
                       ])
    print("Testing DAG-SVM", flush=True)
    pred = np.zeros((X.shape[0]))
    for i, x in enumerate(X):
        x = np.expand_dims(x, 0)
        pred13 = clf_kirc_kich.predict(x)
        if pred13==0:
            pred12 = clf_kirc_kirp.predict(x)
            if pred12==0:
                pred[i] = 0
            else:
                pred[i] = 1
        else:
            pred23 = clf_kirp_kich.predict(x)
            if pred23==0:
                pred[i] = 1
            else:
                pred[i] = 2

    report = classification_report(Y, pred, target_names=["KICH", "KIRC", "KIRP"], output_dict=True, digits=4)
    print(report, flush=True)
    conf_mat = metrics.confusion_matrix(Y, pred)
    print("Confusion Matrix:n", conf_mat, flush=True)
    kappa_score = metrics.cohen_kappa_score(Y, pred)
    print("Cohen Kappa Score = {}".format(kappa_score), flush=True)


def main():
    args = parser.parse_args()
    kirc_train_file = args.kirc_train_file
    kirp_train_file = args.kirp_train_file
    kich_train_file = args.kich_train_file

    kirc_valid_file = args.kirc_valid_file
    kirp_valid_file = args.kirp_valid_file
    kich_valid_file = args.kich_valid_file

    train(kirc_train_file, kirp_train_file, kich_train_file)
    test(kirc_valid_file, kirp_valid_file, kich_valid_file)


if __name__=="__main__":
    main()
