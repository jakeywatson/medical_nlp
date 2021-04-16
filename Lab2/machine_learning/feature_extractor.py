import os
import sys

from machine_learning import Analyzer
from machine_learning.data_handling import *


def extract_features(id_e1, id_e2, entsToNodes, entities, analysis):
    graph = get_tree_graph(analysis)

    lemmas_before, lemmas_between, lemmas_after = get_words_in_sentence(id_e1, id_e2, entities, analysis)
    betweenrels, words_between = find_dependencies_between(id_e1, id_e2, entities, entsToNodes, analysis, graph)
    rels, VB_NN = find_shared_dependencies(id_e1, id_e2, entsToNodes, analysis, graph)
    _2under1 = second_entity_under_first(id_e1, id_e2, entsToNodes, graph)
    preposition = preposition_in_phrase(id_e1, id_e2, entities, entsToNodes, analysis, graph)

    # Adding to feature vector
    features = []

    features.append("2under1=" + str(_2under1))
    features.append("preposition=" + str(preposition))

    if lemmas_before:
        for i in range(0, len(lemmas_before)):
            features.append("lemmas_before=" + lemmas_before[i])

    if lemmas_between:
        for i in range(0, len(lemmas_between)):
            features.append("lemmas_between=" + lemmas_between[i])

    if lemmas_after:
        for i in range(0, len(lemmas_after)):
            features.append("lemmas_after=" + lemmas_after[i])

    if rels:
        for rel in rels:
            features.append("rels="+rel)

    if VB_NN:
        for word in VB_NN:
            features.append("shared_word="+word)

    if betweenrels:
        for i in range(0, len(betweenrels)):
            features.append("rels_between=" + betweenrels[i])
    return features


def output_features(sid, id_e1, id_e2, ddi_type, features, outputfile):
    print("|".join([sid, id_e1, id_e2, ddi_type, *features]), file=outputfile)


def generate_features():

    inputdir = r"..\data\data\Train"
    outfile_name = 'features.txt'
    outputfile = open(outfile_name, 'w+')

    count = 1
    n_files = len(os.listdir(inputdir))

    for f in os.listdir(inputdir):
        sys.stdout.write("\rConsidering file " + str(count) + "/" + str(n_files) + ": " + str(f) + "\n")
        sys.stdout.flush()
        count += 1

        tree = Analyzer.parse(str(inputdir) + "\\" + str(f))

        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            sid = s.attributes["id"].value
            stext = s.attributes["text"].value

            entities = {}
            ents = s.getElementsByTagName("entity")
            for e in ents:
                id = e.attributes["id"].value
                offs = e.attributes["charOffset"].value.split(";")[0].split("-")
                offs.append(e.attributes["text"].value.casefold().strip())
                offs.append(e.attributes["type"].value.casefold().strip())
                entities[id] = offs

            analysis = Analyzer.analyze(stext)

            if analysis:
                entsToNodes = find_tree_ids(entities, analysis)

                pairs = s.getElementsByTagName("pair")
                for pair in pairs:
                    id_e1 = pair.attributes["e1"].value
                    id_e2 = pair.attributes["e2"].value
                    if pair.attributes["ddi"].value == "true" and pair.hasAttribute("type"):
                        ddi = pair.attributes["type"].value
                    else:
                        ddi = "null"

                    features = extract_features(id_e1, id_e2, entsToNodes, entities, analysis)
                    output_features(sid, id_e1, id_e2, ddi, features, outputfile)
    outputfile.close()


