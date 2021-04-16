import subprocess

from machine_learning.check_interaction import *
from machine_learning.learner import trainMegamMaxEnt, trainNLTKMegam


def evaluate(inputdir, outfile_name):
    subprocess.call("java -jar ..\data\eval\evaluateDDI.jar " + inputdir + " " + outfile_name)


def main():
    inputdir = r"..\data\data\Devel"
    outfile_name = 'task9.2ML_Jake_Devel.txt'
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

                    (is_ddi, ddi_type) = check_interaction(id_e1, id_e2, entsToNodes, entities, analysis)
                    print("|".join([sid, id_e1, id_e2, str(is_ddi), ddi_type]), file=outputfile)

    outputfile.close()
    evaluate(inputdir, outfile_name)


if __name__ == '__main__':
    generate_features()
    trainMegamMaxEnt()
    #trainNLTKMegam()
    main()
