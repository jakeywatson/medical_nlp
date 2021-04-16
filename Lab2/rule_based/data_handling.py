import copy

import networkx as nx
import operator

from rule_based.Analyzer import analyze

ROOT = 1


# ----------------------------------------------------------------------------------------------
# WORD MANIPULATION METHODS
# ----------------------------------------------------------------------------------------------

# These methods search areas of the sentence for specific keywords.
# They search the sentence before, between and after the entity window,
# or only the specific that join two entities,


# Count words in a list that match each keywords type - return type with max frequency
def frequent_keyword_type(words, keywords):
    types = {"int": 0, "advise": 0, "effect": 0, "mechanism": 0}
    found = False
    for key in keywords.keys():
        for (word, tag) in words:
            if (word, tag) in keywords[key]:
                types[key] += 1
                found = True
    if found:
        return max(types.items(), key=operator.itemgetter(1))[0]

    return False


# Get words by VERB position regarding entities: before, between, after
# Stores the lemma of the words with POS tags of Verb, Conjunction, Preposition, or Adjective
def get_words_in_sentence(id_e1, id_e2, entities, analysis):
    window_start = int(min(entities[id_e1][0], entities[id_e2][0]))
    window_end = int(max(entities[id_e1][1], entities[id_e2][1]))

    words_before = []
    words_between = []
    words_after = []

    for i in range(1, len(analysis.nodes)):
        if 'start' in analysis.nodes[i]:
            start = int(analysis.nodes[i]["start"])
            end = int(analysis.nodes[i]["end"])
            tag = analysis.nodes[i]["tag"][:2]

            allowed_tag = tag in ["VB", "CC", "IN", "JJ"]

            if allowed_tag:
                if end < window_start:
                    words_before.append((analysis.nodes[i]["lemma"], tag))
                elif start >= window_start and end <= window_end:
                    words_between.append((analysis.nodes[i]["lemma"], tag))
                elif start > window_end:
                    words_after.append((analysis.nodes[i]["lemma"], tag))

    return words_before, words_between, words_after


# Search keywords for a word - without POS tag specification
def keyword_search(word, pos, keywords):
    if pos == "any":
        for key in keywords.keys():
            if any(word in tupl for tupl in keywords[key]):
                return key
    else:
        search_tuple = (word, pos[-2:])
        for key in keywords.keys():
            if search_tuple in keywords[key]:
                return key
    return False


# Search before, after or between the entities for known verbs.
# Searches based on known position of word regarding the entities.
def shared_verb_keyword_search(id_e1, id_e2, entities, entsToNodes, analysis, graph, clues_before, clues_between,
                               clues_after):
    ddi_type = False
    common_verb = entities_under_same_verb(id_e1, id_e2, entsToNodes, analysis, graph)

    if common_verb:
        window_start = int(min(entities[id_e1][0], entities[id_e2][0]))
        window_end = int(max(entities[id_e1][1], entities[id_e2][1]))

        verb_start = int(analysis.nodes[common_verb]["start"])
        verb_end = int(analysis.nodes[common_verb]["end"])
        lemma = analysis.nodes[common_verb]["lemma"]

        # Verb before entities
        if verb_start < window_start:
            ddi_type = keyword_search(lemma, "VB", clues_before)
        # Verb between entities
        elif verb_start >= window_start and verb_end <= window_end:
            ddi_type = keyword_search(lemma, "VB", clues_between)
        # Verb after entities
        elif verb_start > window_end:
            ddi_type = keyword_search(lemma, "VB", clues_after)

    return ddi_type

# Searches the shared path between the root and the entities for the keywords, and returns the ddi type according to the
# keyword.
def path_keyword_search(id_e1, id_e2, pos, entities, entsToNodes, analysis, graph, clues_before, clues_between,
                               clues_after):
    ddi_type = False
    shared_nodes = find_shared_nodes(id_e1, id_e2, entsToNodes, analysis, graph)

    if shared_nodes:
        for node in shared_nodes:
            window_start = int(min(entities[id_e1][0], entities[id_e2][0]))
            window_end = int(max(entities[id_e1][1], entities[id_e2][1]))

            word_start = int(analysis.nodes[node]["start"])
            word_end = int(analysis.nodes[node]["end"])
            lemma = analysis.nodes[node]["lemma"]

            # word before entities
            if word_start < window_start:
                ddi_type = keyword_search(lemma, pos, clues_before)
            # word between entities
            elif word_start >= window_start and word_end <= window_end:
                ddi_type = keyword_search(lemma, pos, clues_between)
            # Verb after entities
            elif word_start > window_end:
                ddi_type = keyword_search(lemma, pos, clues_after)

            if ddi_type:
                return ddi_type
    return ddi_type


# Search before, after or between the entities for known keywords of type (VB, IN, CC, JJ).
# Prioritizes results from between, then before, then after the entities.
# Returns the type of interaction if known, else returns False
def whole_sentence_keyword_search(id_e1, id_e2, entities, analysis, clues_before, clues_between, clues_after):
    before, between, after = get_words_in_sentence(id_e1, id_e2, entities, analysis)

    found_before = frequent_keyword_type(before, clues_before)
    found_between = frequent_keyword_type(between, clues_between)
    found_after = frequent_keyword_type(after, clues_after)

    if found_between:
        return found_between
    elif found_before:
        return found_before
    elif found_after:
        return found_after
    else:
        return False


# Detects multiword entities in the tree using a sliding window approach.
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
            known_id = in_entities(tmp, start, end, entities)
            nodes.append(tokens[i + j][3])
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


# Generates a NetworkX directed graph from the dependency tree generated by CoreNLP.
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
def find_shared_dependencies(id_e1, id_e2, entsToNodes, analysis, graph):
    tree_id_e1 = entsToNodes[id_e1]
    tree_id_e2 = entsToNodes[id_e2]

    root_to_e1 = path_between_entities(ROOT, tree_id_e1, graph)
    root_to_e2 = path_between_entities(ROOT, tree_id_e2, graph)

    nodes = list(i for i in root_to_e1 if i in root_to_e2
                 and analysis.nodes[i]["tag"][:2] in ["VB", "NN", "CC", "JJ", "IN"])
    triples = []

    for node in nodes:
        triples.append((analysis.nodes[node]["head"], node, analysis.nodes[node]["rel"]))
    triples.sort(key=lambda triple: triple[0])

    if len(triples) > 0:
        return triples
    else:
        return False


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


# Finds the relations between the entities
def find_dependencies_between(id_e1, id_e2, entities, entsToNodes, analysis, graph):
    rels = []

    if entities_under_same_verb(id_e1, id_e2, entsToNodes, analysis, graph):
        tree_id_e1 = entsToNodes[id_e1]
        tree_id_e2 = entsToNodes[id_e2]

        noun_e1 = find_noun(tree_id_e1, analysis, graph)
        noun_e2 = find_noun(tree_id_e2, analysis, graph)

        try:
            path = path_between_entities(noun_e1, noun_e2, graph)
            for node in path:
                rels.append(analysis.nodes[node]["rel"])
        except Exception:
            print(analysis)
            print(entities[id_e1][2], id_e1, tree_id_e1, noun_e1)
            print(entities[id_e2][2], id_e2, tree_id_e2, noun_e2)

    return rels


# Finds the verbs which both entities are below in the dependency tree.
def find_entity_verbs(e_id, entsToNodes, analysis, graph):
    tree_id = entsToNodes[e_id]

    root_to_entity = path_between_entities(ROOT, tree_id, graph)
    verbs = list(i for i in root_to_entity if analysis.nodes[i]["tag"][:2] == "VB")

    verbs.sort(key=lambda verb: len(path_between_entities(ROOT, verb, graph)), reverse=True)
    return verbs


# Returns verb if share the same closest verb in the tree.
def entities_under_same_verb(id_e1, id_e2, entsToNodes, analysis, graph):
    e1_verbs = find_entity_verbs(id_e1, entsToNodes, analysis, graph)
    e2_verbs = find_entity_verbs(id_e2, entsToNodes, analysis, graph)

    if len(e1_verbs) == 0 or len(e2_verbs) == 0:
        return False

    if e1_verbs[0] == e2_verbs[0]:
        return e1_verbs[0]
    else:
        return False


# Returns true if the second entity is below the first in the dependency tree
def second_entity_under_first(id_e1, id_e2, entsToNodes, graph):
    tree_id_e1 = entsToNodes[id_e1]
    tree_id_e2 = entsToNodes[id_e2]

    root_to_e1 = path_between_entities(ROOT, tree_id_e1, graph)
    root_to_e2 = path_between_entities(ROOT, tree_id_e2, graph)

    if tree_id_e2 in root_to_e1:
        return False
    elif tree_id_e1 in root_to_e2:
        return True
    else:
        return False


# ----------------------------------------------------------------------------------------------
# PHRASE SEARCH METHODS
# ----------------------------------------------------------------------------------------------
# These methods provide means to search for specific words or word types in the phrase.
# They check if the phrase contains a preposition, if there is a negative conjunction
# in the phrase and between the entities, and if there is a negation operator in the phrase.

def check_dependencies_between(id_e1, id_e2, entities, entsToNodes, analysis, graph):
    rel_types = {
        "mechanism": {"nsubj", "obj", "nsubjpass"},
        "effect": {"nsubj", "nmod", "nsubjpass"},
        "advise": {"nsubjpass", "nmod", "obj"},
        "int": {"nmod", "nsubj"}
    }

    shared_verb = entities_under_same_verb(id_e1, id_e2, entsToNodes, analysis, graph)
    if shared_verb:
        rels = find_dependencies_between(id_e1, id_e2, entities, entsToNodes, analysis, graph)
        for rel in rels:
            for rel_type in ["advise", "effect", "mechanism", "int"]:
                if rel in rel_types[rel_type]:
                    return rel_type

    return False


# Checks for prepositions in the phrase
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


# Checks for negative conjunctions in the phrase, returns true one is found and it is
# between the entities in the sentence.
def negative_conjunction_between_entities(id_e1, id_e2, entities, entsToNodes, analysis, graph):
    neg_conjunctions = {"nor", "or", "neither"}

    tree_id_e1 = entsToNodes[id_e1]
    tree_id_e2 = entsToNodes[id_e2]

    window_start = int(min(entities[id_e1][0], entities[id_e2][0]))
    window_end = int(max(entities[id_e1][1], entities[id_e2][1]))

    e1_noun = find_noun(tree_id_e1, analysis, graph)
    e2_noun = find_noun(tree_id_e2, analysis, graph)

    try:
        e1_noun_deps = nx.dfs_successors(graph, e1_noun, depth_limit=1)
        e2_noun_deps = nx.dfs_successors(graph, e2_noun, depth_limit=1)
    except KeyError:
        print(analysis)
        print(entities)
        print(sliding_window(entities, analysis))
        print(entities[id_e1][2], id_e1, tree_id_e1, e1_noun)
        print(entities[id_e2][2], id_e2, tree_id_e2, e2_noun)
        raise Exception("No noun found")

    for node in e1_noun_deps.keys():
        for i in e1_noun_deps[node]:
            if 'start' in analysis.nodes[i]:
                start = int(analysis.nodes[i]["start"])
                end = int(analysis.nodes[i]["end"])
                tag = analysis.nodes[i]["tag"]
                lemma = analysis.nodes[i]["lemma"]

                between_entities = start >= window_start and end <= window_end
                is_negative_conjunction = "CC" in tag and lemma in neg_conjunctions

                if between_entities and is_negative_conjunction:
                    return True

    for node in e2_noun_deps.keys():
        for i in e2_noun_deps[node]:
            if 'start' in analysis.nodes[i]:
                start = int(analysis.nodes[i]["start"])
                end = int(analysis.nodes[i]["end"])
                tag = analysis.nodes[i]["tag"]
                lemma = analysis.nodes[i]["lemma"]

                between_entities = start >= window_start and end <= window_end
                is_negative_conjunction = "CC" in tag and lemma in neg_conjunctions

                if between_entities and is_negative_conjunction:
                    return True
    return False


# Checks for negation words in the phrase or under the shared verb, returns true one is found.
def negation_in_phrase(id_e1, id_e2, entsToNodes, analysis, graph):
    negations = {"""ain't""", """can't""", """didn't""", """don't""", """isn't""", """mustnt""", """never""",
                 """nobody""", """not""", """shouldn't""", """weren't""", """wouldn't""", """won't""", """wasn't""",
                 """shan't""", """nor""", """mightn't""", """hasn't""", """doesn't""", """couldn't""", """aren't""",
                 """no"""}

    common_verb = entities_under_same_verb(id_e1, id_e2, entsToNodes, analysis, graph)
    if common_verb:
        tree_id_e1 = entsToNodes[id_e1]
        tree_id_e2 = entsToNodes[id_e2]

        e1_to_verb = nx.shortest_path(graph, common_verb, tree_id_e1)
        e2_to_verb = nx.shortest_path(graph, common_verb, tree_id_e2)
        verb_deps = nx.dfs_successors(graph, common_verb, depth_limit=1)

        for node in e1_to_verb:
            if analysis.nodes[node]["lemma"] in negations:
                return True
        for node in e2_to_verb:
            if analysis.nodes[node]["lemma"] in negations:
                return True
        for node in verb_deps[common_verb]:
            head = analysis.nodes[node]["head"]
            if analysis.nodes[node]["lemma"] in negations \
                    and head == common_verb:
                return True
    return False


def read_training_data():
    clue_words_after = {"int": set(), "effect": set(), "mechanism": set(), "advise": set()}
    clue_words_before = {"int": set(), "effect": set(), "mechanism": set(), "advise": set()}
    clue_words_between = {"int": set(), "effect": set(), "mechanism": set(), "advise": set()}

    before_file = open("clues_before.txt")
    for line in before_file.read().splitlines():
        split = line.split(":")
        i_type = split[0]
        word, tag = split[1].split(",")
        clue_words_before[i_type].add((word, tag))
    before_file.close()

    between_file = open("clues_between.txt")
    for line in between_file.read().splitlines():
        split = line.split(":")
        i_type = split[0]
        word, tag = split[1].split(",")
        clue_words_between[i_type].add((word, tag))
    between_file.close()

    after_file = open("clues_after.txt")
    for line in after_file.read().splitlines():
        split = line.split(":")
        i_type = split[0]
        word, tag = split[1].split(",")
        clue_words_after[i_type].add((word, tag))
    after_file.close()

    return clue_words_before, clue_words_between, clue_words_after


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

# test = "However, administration of 4-methylpyrazole (90 mg kg(-1) body weight) to rats 2 hours prior to 1,3-difluoroacetone (100 mg kg(-1) body weight) was ineffective in preventing (-)-erythro-fluorocitrate synthesis and did not diminish fluoride or citrate accumulation in vivo progestin."
# entities = {"one":[27, 42, "4-methylpyrazole"],
#             "two":[96, 114, "1,3-difluoroacetone"],
#             "three":[174, 198, "(-)-erythro-fluorocitrate"]}

# test = "[The GABA-ergic system and brain edema]&#xd;&#xa;It has been shown in rats with experimental toxic and traumatic edemas that picrotoxin (1 mg/kg) removes the antiedematous action of diazepam, phenazepam, phenibut and amizyl and reduces the action of phentolamine. "
# entities = {"zero": [117, 126, "picrotoxin"],
#             "one": [174, 181, "diazepam"],
#             "two": [184, 193, "phenazepam"],
#             "three": [196, 203, "phenibut"],
#             "four": [209, 214, "amizyl"],
#             "five": [242, 253, "phentolamine"],
#             }
#
# test = "There is, however, currently no data on the effect of combined hyperglycaemia and hyperinsulinaemia on the renal and ocular blood flow seen in diabetic patients on insulin therapy. "
# entities = {"zero": [87, 93, "insulin"]
#             }
#
# test =""
# entities = {}
#
# c_bef, c_bet, c_aft = read_training_data()
#
# parsed = test.replace("&#xd;&#xa;", "  ")
# analysis = analyze(parsed)
#
# graph = get_tree_graph(analysis)
# entsToNodes = find_tree_ids(entities, analysis)
#
# print(analysis)
# print(sliding_window(entities, analysis))
# print(entities)
#
# id1 = entsToNodes["one"]
# id2 = entsToNodes["two"]
# id3 = entsToNodes["three"]
# #
# one, two, three = "one", "two", "three"
# print(id1, id2, id3)

# # ----------------------------------------------------------------------------------------------
# # WORD MANIPULATION TESTS
# # ----------------------------------------------------------------------------------------------
#
# bef, bet, aft = get_words_in_sentence(one, two, entities, analysis)
# print(shared_verb_keyword_search(one, two, entities, entsToNodes, analysis, graph, c_bef, c_bet, c_aft))
# print(path_keyword_search(one, two, "any", entities, entsToNodes, analysis, graph, c_bef, c_bet, c_aft))
# print(path_keyword_search(one, two, "VB", entities, entsToNodes, analysis, graph, c_bef, c_bet, c_aft))
# print(path_keyword_search(one, two, "NN", entities, entsToNodes, analysis, graph, c_bef, c_bet, c_aft))
# print(path_keyword_search(one, two, "JJ", entities, entsToNodes, analysis, graph, c_bef, c_bet, c_aft))
# print(path_keyword_search(one, two, "CC", entities, entsToNodes, analysis, graph, c_bef, c_bet, c_aft))
#
# print(whole_sentence_keyword_search(one, two, entities, analysis, c_bef, c_bet, c_aft))
# print(sliding_window(entities, analysis))
#
# # ----------------------------------------------------------------------------------------------
# # PHRASE PROPERTIES TESTS
# # ----------------------------------------------------------------------------------------------
#
# print(find_entity_verbs(one, entsToNodes, analysis, graph))
# print(find_entity_verbs(two, entsToNodes, analysis, graph))
# print(find_shared_dependencies(one, two, entsToNodes, analysis, graph))
# print(find_shared_nodes(one, two, entsToNodes, analysis, graph))
# print(second_entity_under_first(one, two, entsToNodes, graph))
# print(entities_under_same_verb(one, two, entsToNodes, analysis, graph))
# print(find_dependencies_between(one, two, entities, entsToNodes, analysis, graph))
#
# # ----------------------------------------------------------------------------------------------
# # PHRASE SEARCH TESTS
# # ----------------------------------------------------------------------------------------------
#
# print(check_dependencies_between(one, two, entities, entsToNodes, analysis, graph))
# print(preposition_in_phrase(one, two, entities, entsToNodes, analysis, graph))
# print(negative_conjunction_between_entities(one, two, entities, entsToNodes, analysis, graph))
# print(negation_in_phrase(one, two, entsToNodes, analysis, graph))
