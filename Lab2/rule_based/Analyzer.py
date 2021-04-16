from xml.dom.minidom import parseString

from nltk.parse.corenlp import CoreNLPDependencyParser

parser = CoreNLPDependencyParser(url="http://localhost:9000")


def parse(filename):
    xml_friendly = open(filename).read().replace("&#xd;&#xa;", "  ")
    return parseString(xml_friendly)


def analyze(s):
    current_position = 0
    if s:
        s = s.strip()
        dep_graph, = parser.raw_parse(s)
        for i in range(1, len(dep_graph.nodes)):
            node = dep_graph.nodes[i]
            word = node["word"]

            start = current_position + s[current_position:].find(word)
            end = start + len(word) - 1

            current_position = end

            if start != -1:
                node["start"] = start
                node["end"] = end

        return dep_graph
    else:
        return None
