import os
import sys

from machine_learning import Analyzer


def test():

    inputdir = r"..\data\data\Train"

    true_counter = 0
    false_counter = 0

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
            pairs = s.getElementsByTagName("pair")
            for pair in pairs:
                if pair.attributes["ddi"].value == "true" and pair.hasAttribute("type"):
                    true_counter += 1
                else:
                    false_counter += 1
    print(false_counter)
    print(true_counter)

test()