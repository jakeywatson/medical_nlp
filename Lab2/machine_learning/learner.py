import os
import pickle

import nltk
from nltk.classify import MaxentClassifier


def read_features(features_filename):
    featurefile = open(features_filename, "r")
    combined = []
    for line in featurefile.read().splitlines():
        labels = []
        features = []
        split = line.split("|")
        for elem in split[:4]:
            labels.append(elem)
        for elem in split[4:]:
            features.append(elem)
        row = (labels[3], features)
        combined.append(row)
    featurefile.close()
    return combined


def trainMegamMaxEnt():
    print("Loading features...")
    features = read_features("features.txt")
    features2Megam = open("megam_input.txt", "w+")

    for row in features:
        print(row[0], *row[1], sep=" ", file=features2Megam)
    features2Megam.close()
    print("Training model")
    megam_model = "megam_model.dat"
    os.system("megam -nc -nobias -repeat 10 -maxi 1000 -dpp 0.00000001 multiclass megam_input.txt > " + megam_model)


def trainNLTKMegam():
    print("Loading features...")
    features = read_features("features.txt")

    for i in range(0, len(features)):
        label = features[i][0]
        feats = dict([(f, True) for f in features[i][1]])
        features[i] = (feats, label)

    print("Training model")
    nltk.classify.config_megam(r"C:\Users\jaked\OneDrive\Documents\megam_0.92\megam.exe")
    classifier = MaxentClassifier.train(features, algorithm="megam", max_iter=500, min_ll=0.00000001)
    save_classifier = open("nltk_megam.pickle", "wb+")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()


def read_model(modelfile):
    classifierfile = open(modelfile, "rb")
    classifier = pickle.load(classifierfile)
    classifierfile.close()
    return classifier


