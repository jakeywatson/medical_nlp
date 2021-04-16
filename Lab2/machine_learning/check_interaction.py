from machine_learning.feature_extractor import *
from machine_learning.learner import read_model
from machine_learning.model import MEmodel


megam = MEmodel("megam_model.dat")
megam_nltk = read_model("nltk_megam.pickle")


def check_interaction(id_e1, id_e2, entsToNodes, entities, analysis):
    features = extract_features(id_e1, id_e2, entsToNodes, entities, analysis)
    dist = megam.prob_dist_z(features)
    best = None
    mx = 0
    for c in dist:
        if dist[c] > mx:
            mx = dist[c]
            best = c
    if best == "null":
        return 0, "null"
    else:
        return 1, best


def check_interaction_NLTK(id_e1, id_e2, entsToNodes, entities, analysis):
    classifier = megam_nltk
    features = extract_features(id_e1, id_e2, entsToNodes, entities, analysis)
    features = dict([(f, True) for f in features])

    label = classifier.classify(features)

    if label == "null":
        return 0, "null"
    else:
        return 1, label
