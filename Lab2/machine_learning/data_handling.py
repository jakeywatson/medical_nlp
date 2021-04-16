import copy

import networkx as nx

# Root of the tree
ROOT = 1

# ----------------------------------------------------------------------------------------------
# WORD MANIPULATION METHODS
# ----------------------------------------------------------------------------------------------

# These methods search areas of the sentence for specific keywords.
# They search the sentence before, between and after the entity window,
# or only the specific that join two entities,


# Get words by VERB position regarding entities: before, between, after
# Stores the lemma of the words with POS tags of Verb, Conjunction, Preposition, or Adjective
# Filters out the entity names.
def get_words_in_sentence(id_e1, id_e2, entities, analysis):
    window_start = int(min(entities[id_e1][0], entities[id_e2][0]))
    window_end = int(max(entities[id_e1][1], entities[id_e2][1]))

    words_before = []
    words_between = []
    words_after = []

    starts = []
    for entity in entities:
        e_start = int(entities[entity][0])
        e_end = int(entities[entity][1])
        for i in range(1, len(analysis.nodes) - 1):
            start = int(analysis.nodes[i]["start"])
            end = int(analysis.nodes[i]["end"])
            if start >= e_start and end <= e_end:
                starts.append(int(analysis.nodes[i]["start"]))

    for i in range(1, len(analysis.nodes)):
        if 'start' in analysis.nodes[i]:
            word = analysis.nodes[i]["lemma"]
            start = int(analysis.nodes[i]["start"])
            end = int(analysis.nodes[i]["end"])
            tag = analysis.nodes[i]["tag"][:2]

            allowed_tag = tag in ["CC", "IN", "JJ", "NN"]

            if allowed_tag and int(analysis.nodes[i]["start"]) not in starts:
                if end < window_start:
                    words_before.append(analysis.nodes[i]["lemma"])
                elif start >= window_start and end <= window_end:
                    words_between.append(analysis.nodes[i]["lemma"])
                elif start > window_end:
                    words_after.append(analysis.nodes[i]["lemma"])

    return words_before, words_between, words_after


# Detect multi-word entities in the tree, and returns the set of nodes associated to each one.
# Maximum node-length of entity that can be detected - 10
def sliding_window(entities, analysis):
    def in_entities(word, start, end, entities):
        search_term = word.strip()
        if search_term[-2:] in ['A.', 'B.', 'D.', 'E.', 'S.']:
            search_term = search_term[:-1]

        search_term = search_term.casefold().strip()

        for e_id in entities.keys():
            known = entities[e_id][2].casefold().strip()
            e_start = entities[e_id][0]
            e_end = entities[e_id][1]

            between_entity_offsets = (int(start) >= int(e_start) and int(end) <= int(e_end))
            partial_offset_match = int(e_start) == int(start) or int(e_end) == int(end)

            if partial_offset_match:
                return e_id
            else:
                if search_term in known and between_entity_offsets:
                    return e_id
                elif known in search_term:
                    substring_start = start + search_term.find(known)
                    substring_end = substring_start + len(known) - 1
                    substring_between_offsets = int(substring_start) >= int(e_start) and int(substring_end) <= int(e_end)
                    if substring_between_offsets:
                        return e_id

        return False

    tokens = []
    for node in range(1, len(analysis.nodes)):
        if "start" in analysis.nodes[node]:
            start = int(analysis.nodes[node]["start"])
            end = int(analysis.nodes[node]["end"])
            word = analysis.nodes[node]["word"]
            rel = analysis.nodes[node]["rel"]
            tag = analysis.nodes[node]["tag"]

            node_index = node
            tokens.append([start, end, word, node_index, rel, tag])

    found = {}
    cut_tokens = copy.deepcopy(tokens)

    for i in range(0, len(tokens) - 1):
        tmp = tokens[i][2]
        start = tokens[i][0]
        end = tokens[i][1]
        nodes = [tokens[i][3]]
        rel = tokens[i][4]
        tag = tokens[i][5]

        known_id = in_entities(tmp, start, end, entities)
        if in_entities(tmp, start, end, entities):
            cut_tokens[i] = None
            found[known_id] = nodes
            continue

        for j in range(1, 10):
            if i + j == len(tokens):
                break

            if tokens[i + j][4] == "punct" or rel[-5:] == "punct" \
                    or tokens[i + j][5] == "SYM" or tokens[i + j - 1][5] == "SYM":
                tmp = tmp + "" + tokens[i + j][2]
            else:
                tmp = tmp + " " + tokens[i + j][2]
            end = tokens[i + j][1]
            rel = rel + tokens[i + j][4]
            nodes.append(tokens[i + j][3])
            known_id = in_entities(tmp, start, end, entities)
            if known_id:
                for x in range(i, i + j + 1):
                    cut_tokens[x] = None
                found[known_id] = nodes
                break

    return found


# ----------------------------------------------------------------------------------------------
# TREE MANIPULATION METHODS
# ----------------------------------------------------------------------------------------------
# These methods turn the dependency tree generated by CoreNLP into a NetworkX directed graph,
# which has utilities for exploiting relationships between nodes.

# The methods provide an index matcher for entity IDs and node IDs, a graph generator, and a method
# to find the shortest path between nodes.


# Converts entity IDs to tree node numbers
# Matches entities to nodes using a sliding window, which matches using offset positions and word content.
def find_tree_ids(entities, analysis):
    tree_ids = {}
    multi_word_identifiers = sliding_window(entities, analysis)

    for e_id in entities.keys():
        found = False
        e_start = entities[e_id][0]
        e_end = entities[e_id][1]

        # Exact match on both offsets
        for i in range(1, len(analysis.nodes)):
            if 'start' in analysis.nodes[i]:
                if int(e_start) == int(analysis.nodes[i]["start"]) and int(e_end) == int(analysis.nodes[i]["end"]):
                    tree_ids[e_id] = int(i)
                    found = True
                    break

        # Detects multi-word entities in the tree and links them to their entity IDs
        if e_id in multi_word_identifiers:
            for node in multi_word_identifiers[e_id]:
                if analysis.nodes[node]["tag"][0:2] == "NN" and analysis.nodes[node]["rel"] != "compound":
                    tree_ids[e_id] = node
                    found = True
                    break
            # If no noun detected, give the last node in the list of nodes associated to the entity
            tree_ids[e_id] = multi_word_identifiers[e_id][-1]
            found = True

        if not found:
            # If can't find entity in the tree - throw exception, print details
            print(analysis)
            print(entities)
            print(sliding_window(entities, analysis))
            sentence = ""
            for i in range(1, len(analysis.nodes)):
                sentence = sentence + " " + analysis.nodes[i]["word"]
            print(sentence)
            raise Exception("NO TREE ID FOUND FOR ENTITY: " + e_id, e_start, e_end)

    return tree_ids


# Generates a NetworkX graph from the dependency tree generated by CoreNLP.
def get_tree_graph(analysis):
    def get_deps(analysis):
        deps = {}
        for i in range(1, len(analysis.nodes)):
            deps[i] = [item for sublist in list(analysis.nodes[i]["deps"].values()) for item in sublist]
        return deps

    def deps_to_edges(deps):
        edges = []
        for src in deps.keys():
            for dst in deps[src]:
                edges.append([src, dst])
        final = []
        for i in range(0, len(edges)):
            already_present = False
            for j in range(0, len(final)):
                if final[j][0] == edges[i][0] and final[j][1] == edges[j][1]:
                    already_present = True
                    break
            if not already_present:
                final.append(edges[i])
        return edges

    edges = deps_to_edges(get_deps(analysis))
    graph = nx.Graph(edges)
    return graph


# Finds the shortest path between two nodes of the graph
def path_between_entities(id_e1, id_e2, graph):
    try:
        return nx.shortest_path(graph, source=id_e1, target=id_e2)
    except nx.exception.NetworkXNoPath:
        raise Exception("Exception: No path between nodes" + str(id_e1) + ", " + str(id_e2))
    except nx.exception.NodeNotFound:
        raise Exception("Exception: node not found, either " + str(id_e1) + ", " + str(id_e2))


# ----------------------------------------------------------------------------------------------
# PHRASE PROPERTIES METHODS
# ----------------------------------------------------------------------------------------------

# These methods provide properties of the phrase containing both entities. The phrase is usually
# defined here as the words dependent on the verb connecting both entities.


# Finds the non-compound noun connected to the entity
def find_noun(id_e1, analysis, graph):
    root_to_entity = path_between_entities(id_e1, ROOT, graph)
    for node in root_to_entity:
        if analysis.nodes[node]["tag"][0:2] == "NN" and analysis.nodes[node]["rel"] != "compound":
            return node

    for node in root_to_entity:
        if analysis.nodes[node]["tag"][0:2] == "JJ":
            return node

    return id_e1


# Finds the words which both entities are below in the dependency tree.
def find_shared_nodes(id_e1, id_e2, entsToNodes, analysis, graph):
    tree_id_e1 = entsToNodes[id_e1]
    tree_id_e2 = entsToNodes[id_e2]

    root_to_e1 = path_between_entities(ROOT, tree_id_e1, graph)
    root_to_e2 = path_between_entities(ROOT, tree_id_e2, graph)

    nodes = list(i for i in root_to_e1 if i in root_to_e2
                 and analysis.nodes[i]["tag"][:2] in ["VB", "NN", "CC", "JJ", "IN"])

    if len(nodes) > 0:
        return nodes
    else:
        return False


# Find the words that both entities depend on, all the way to the root of the tree.
# Only returns the entities that are in both paths, and that are either nouns or verbs.
# Also returns the relation of the word to its head, if the word is the head of an entity.
def find_shared_dependencies(id_e1, id_e2, entsToNodes, analysis, graph):
    shared_path = find_shared_nodes(id_e1, id_e2, entsToNodes, analysis, graph)
    tree_id1 = entsToNodes[id_e1]
    tree_id2 = entsToNodes[id_e2]

    rels = []
    VBs_NNs = []

    for node in shared_path:
        if analysis.nodes[node]["tag"][:2] in ["VB", "NN"]\
                and (node != tree_id1 and node != tree_id2):
            if analysis.nodes[tree_id1]["head"] == node:
                rels.append(analysis.nodes[tree_id1]["rel"])
            if analysis.nodes[tree_id2]["head"] == node:
                rels.append(analysis.nodes[tree_id2]["rel"])
            VBs_NNs.append(
                analysis.nodes[node]["lemma"] + "-" + analysis.nodes[node]["tag"][:2]
            )
    return rels, VBs_NNs


# Finds the words and relations between the entities. Only returns
# verbs, conjunctions, preposition, or adjectives.
def find_dependencies_between(id_e1, id_e2, entities, entsToNodes, analysis, graph):
    rels_between = []
    words_between = []

    tree_id_e1 = entsToNodes[id_e1]
    tree_id_e2 = entsToNodes[id_e2]

    try:
        path = path_between_entities(tree_id_e1, tree_id_e2, graph)
        for node in path:
            if analysis.nodes[node]["tag"][:2] in ["VB", "CC", "IN", "JJ"]:
                if analysis.nodes[tree_id_e1]["head"] == node:
                    rels_between.append(analysis.nodes[tree_id_e1]["rel"])
                if analysis.nodes[tree_id_e2]["head"] == node:
                    rels_between.append(analysis.nodes[tree_id_e2]["rel"])
                words_between.append(analysis.nodes[node]["lemma"]+"-"+analysis.nodes[node]["tag"][:2])
    except Exception:
        print(analysis)
        print(entities[id_e1][2], id_e1, tree_id_e1)
        print(entities[id_e2][2], id_e2, tree_id_e2)

    return rels_between, words_between


# Finds the set of verbs which both entities are below in the dependency tree.
def find_entity_verbs(e_id, entsToNodes, analysis, graph):
    tree_id = entsToNodes[e_id]

    root_to_entity = path_between_entities(ROOT, tree_id, graph)
    verbs = list(i for i in root_to_entity if analysis.nodes[i]["tag"][:2] == "VB")

    verbs.sort(key=lambda verb: len(path_between_entities(ROOT, verb, graph)), reverse=True)
    return verbs


# Returns true if the second entity is below the first in the dependency tree.
# if neither entity is below the other, returns "neither".
def second_entity_under_first(id_e1, id_e2, entsToNodes, graph):
    tree_id_e1 = entsToNodes[id_e1]
    tree_id_e2 = entsToNodes[id_e2]

    root_to_e1 = path_between_entities(ROOT, tree_id_e1, graph)
    root_to_e2 = path_between_entities(ROOT, tree_id_e2, graph)

    if tree_id_e2 in root_to_e1:
        return False
    elif tree_id_e1 in root_to_e2:
        return True
    return "neither"


# ----------------------------------------------------------------------------------------------
# PHRASE SEARCH METHODS
# ----------------------------------------------------------------------------------------------
# These methods provide means to search for specific words or word types in the phrase.
# They check if the phrase contains a preposition, if there is a negative conjunction
# in the phrase and between the entities, and if there is a negation operator in the phrase.
# Returns verb if share the same closest verb in the tree.


# Returns the closest verb to both entities if it is closest for both entities - otherwise
# returns false.
def entities_under_same_verb(id_e1, id_e2, entsToNodes, analysis, graph):
    e1_verbs = find_entity_verbs(id_e1, entsToNodes, analysis, graph)
    e2_verbs = find_entity_verbs(id_e2, entsToNodes, analysis, graph)

    if len(e1_verbs) == 0 or len(e2_verbs) == 0:
        return False

    if e1_verbs[0] == e2_verbs[0]:
        return e1_verbs[0]
    else:
        return False


# Checks for prepositions in the phrase. It checks if both entities are under the same verb,
# and if they are it looks for a preposition tag in the path between the entities and their shared verb.
# if no preposition found it returns False, if one is found it returns true, and if they do not share a verb,
# it returns "shared_verb"
def preposition_in_phrase(id_e1, id_e2, entities, entsToNodes, analysis, graph):
    common_verb = entities_under_same_verb(id_e1, id_e2, entsToNodes, analysis, graph)
    if common_verb:
        tree_id_e1 = entsToNodes[id_e1]
        tree_id_e2 = entsToNodes[id_e2]

        e1_noun = find_noun(tree_id_e1, analysis, graph)
        e2_noun = find_noun(tree_id_e2, analysis, graph)

        try:
            e1_noun_deps = nx.dfs_successors(graph, e1_noun, depth_limit=1)
            e2_noun_deps = nx.dfs_successors(graph, e2_noun, depth_limit=1)
        except Exception:
            print(analysis)
            print(entities)
            print(sliding_window(entities, analysis))
            print(entities[id_e1][2], id_e1, tree_id_e1, e1_noun)
            print(entities[id_e2][2], id_e2, tree_id_e2, e2_noun)
            raise Exception("No noun found")

        for node in e1_noun_deps.keys():
            for dep_node in e1_noun_deps[node]:
                if analysis.nodes[dep_node]["tag"][:2] == "IN":
                    return True
        for node in e2_noun_deps.keys():
            for dep_node in e2_noun_deps[node]:
                if analysis.nodes[dep_node]["tag"][:2] == "IN":
                    return True
        return False
    return "shared_verb"

# ----------------------------------------------------------------------------------------------
# TESTS SETUP
# series of basic tests to check functionality of the sentence and word analysis methods.
# ----------------------------------------------------------------------------------------------
#
# #       1       2      3  4         5    6         7          8  9         10   11   12      13  14
# test = "Caution should be exercised when combining resorcinol or salicylic acid with DIFFERIN Gel."
# entities = {"one": [43, 53, "resorcinol"],
#             "two": [57, 70, "salicylic acid"],
#             "three": [77, 88, "DIFFERIN Gel"]}


# parsed = test.replace("&#xd;&#xa;", "  ")
# analysis = analyze(parsed)
#
# graph = get_tree_graph(analysis)
# entsToNodes = find_tree_ids(entities, analysis)
#
# print(analysis)
# print(entities)
# print(sliding_window(entities, analysis))
#
# id1 = entsToNodes["one"]
# id2 = entsToNodes["two"]
# id3 = entsToNodes["three"]
#
# one, two, three = "one", "two", "three"
# print(id1, id2, id3)

# # ----------------------------------------------------------------------------------------------
# # WORD MANIPULATION TESTS
# # ----------------------------------------------------------------------------------------------
# #
# bef, bet, aft = get_words_in_sentence(one, two, entities, analysis)
# print(bef)
# print(bet)
# print(aft)
# print(sliding_window(entities, analysis))
#
# # ----------------------------------------------------------------------------------------------
# # PHRASE PROPERTIES TESTS
# # ----------------------------------------------------------------------------------------------
#

# print(find_shared_dependencies(one, two, entsToNodes, analysis, graph))
# print(find_shared_nodes(one, two, entsToNodes, analysis, graph))
# print(second_entity_under_first(one, two, entsToNodes, graph))
# print(entities_under_same_verb(one, two, entsToNodes, analysis, graph))
# print(find_dependencies_between(one, two, entities, entsToNodes, analysis, graph))

# # ----------------------------------------------------------------------------------------------
# # PHRASE SEARCH TESTS
# # ----------------------------------------------------------------------------------------------

# print(preposition_in_phrase(one, two, entities, entsToNodes, analysis, graph))
