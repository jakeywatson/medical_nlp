import os
import sys

from rule_based.Analyzer import parse, analyze
from rule_based.data_handling import get_words_in_sentence


# Dataset information extraction - extracts common keywords from each sentence, and saves the most frequent for each ddi type
# Ensures correctness by removing entity names, removing repeated keywords for different types, except the most frequent type,
# and removes punctuations. Divides them according to position in sentence relative to the entities - before, between and after,
# and saves each in a file.
def get_clue_words():
    def fill_clues(mecs, effs, advs, ints, keywords):
        combined = list(mecs + effs + advs + ints)
        for (word, tag) in combined:
            if combined.count((word, tag)) > 1:

                ints_freq = ints.count((word, tag)) / len(ints)
                advs_freq = advs.count((word, tag)) / len(advs)
                effs_freq = effs.count((word, tag)) / len(effs)
                mecs_freq = mecs.count((word, tag)) / len(mecs)

                maximum = max(ints_freq, advs_freq, effs_freq, mecs_freq)

                if maximum > 0.0001:
                    if maximum == ints_freq:
                        keywords["int"].add((word, tag))
                    elif maximum == advs_freq:
                        keywords["advise"].add((word, tag))
                    elif maximum == mecs_freq:
                        keywords["mechanism"].add((word, tag))
                    elif maximum == effs_freq:
                        keywords["effect"].add((word, tag))

    def remove_drug_names(drug_names, clues):
        for key in clues.keys():
            newset = set()
            for (word, tag) in clues[key]:
                if word.casefold().strip() not in drug_names:
                    newset.add((word, tag))
            clues[key] = newset

    def remove_punctuation(clues):
        punctuations = '''![]{};:'"\,<>./?@#$%^&*_-~()'''
        for key in clues.keys():
            newset = set()
            for (word, tag) in clues[key]:
                if not any(x in punctuations for x in word):
                    newset.add((word, tag))
            clues[key] = newset

    def fill_clue_words(drug_names, words_m, words_e, words_a, words_i, clues):
        fill_clues(words_m, words_e, words_a, words_i, clues)
        remove_drug_names(drug_names, clues)
        remove_punctuation(clues)

    clue_words_after = {"int": set(), "effect": set(), "mechanism": set(), "advise": set()}
    clue_words_before = {"int": set(), "effect": set(), "mechanism": set(), "advise": set()}
    clue_words_between = {"int": set(), "effect": set(), "mechanism": set(), "advise": set()}

    words_before_i = []
    words_after_i = []
    words_between_i = []

    words_before_e = []
    words_after_e = []
    words_between_e = []

    words_before_m = []
    words_after_m = []
    words_between_m = []

    words_before_a = []
    words_after_a = []
    words_between_a = []

    drug_names = set()

    count = 1
    n_files = len(os.listdir("../data/data/Train/"))

    for f in os.listdir("../data/data/Train/"):
        sys.stdout.write("\rConsidering file " + str(count) + "/" + str(n_files) + ": " + str(f) + "\n")
        sys.stdout.flush()
        count += 1

        tree = parse("../data/data/Train/" + str(f))
        sentences = tree.getElementsByTagName("sentence")

        print("memory of words")
        print(len(words_before_m), len(words_before_e), len(words_before_i), len(words_before_a))
        print(len(words_between_m), len(words_between_e), len(words_between_i), len(words_between_a))
        print(len(words_after_m), len(words_after_e), len(words_after_i), len(words_after_a))

        for s in sentences:
            stext = s.attributes["text"].value

            entities = {}
            ents = s.getElementsByTagName("entity")

            for e in ents:
                id = e.attributes["id"].value
                name = e.attributes["text"].value
                offs = e.attributes["charOffset"].value.split(";")[0].split("-")
                entities[id] = offs
                drug_names.add(name.casefold().strip())

            analysis = analyze(stext)
            pairs = s.getElementsByTagName("pair")

            for pair in pairs:
                id_e1 = pair.attributes["e1"].value
                id_e2 = pair.attributes["e2"].value
                ddi = pair.attributes["ddi"].value

                if ddi == "true" and pair.hasAttribute("type"):
                    ddi_type = pair.attributes["type"].value

                    before, between, after = get_words_in_sentence(id_e1, id_e2, entities, analysis)
                    print("new words")
                    print(len(before), len(between), len(after))

                    if ddi_type == "mechanism":
                        words_before_m.extend(before)
                        words_between_m.extend(between)
                        words_after_m.extend(after)

                    elif ddi_type == "effect":
                        words_before_e.extend(before)
                        words_between_e.extend(between)
                        words_after_e.extend(after)

                    elif ddi_type == "advise":
                        words_before_a.extend(before)
                        words_between_a.extend(between)
                        words_after_a.extend(after)

                    elif ddi_type == "int":
                        words_before_i.extend(before)
                        words_between_i.extend(between)
                        words_after_i.extend(after)

    fill_clue_words(drug_names, words_before_m, words_before_e, words_before_a, words_before_i, clue_words_before)
    fill_clue_words(drug_names, words_between_m, words_between_e, words_between_a, words_between_i, clue_words_between)
    fill_clue_words(drug_names, words_after_m, words_after_e, words_after_a, words_after_i, clue_words_after)

    return clue_words_before, clue_words_between, clue_words_after


# IO: Create dataset information

def write_training_data():
    clue_words_before, clue_words_between, clue_words_after = get_clue_words()

    with open('clues_before.txt', 'w+') as before_file:
        for e_type in clue_words_before.keys():
            for (word, tag) in clue_words_before[e_type]:
                before_file.write(e_type + ":" + word + "," + tag + "\n")
    with open('clues_between.txt', 'w+') as between_file:
        for e_type in clue_words_between.keys():
            for (word, tag) in clue_words_between[e_type]:
                between_file.write(e_type + ":" + word + "," + tag + "\n")
    with open('clues_after.txt', 'w+') as after_file:
        for e_type in clue_words_after.keys():
            for (word, tag) in clue_words_after[e_type]:
                after_file.write(e_type + ":" + word + "," + tag + "\n")
    before_file.close()
    between_file.close()
    after_file.close()

