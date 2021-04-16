from rule_based.data_handling import *


def check_interaction(id_e1, id_e2, entities, entsToNodes, analysis, clues_before, clues_between, clues_after):
    ddi_type = False

    graph = get_tree_graph(analysis)

    # ----------------------------------------------------------------------------------------------
    # WORD MANIPULATION TESTS
    # ----------------------------------------------------------------------------------------------

    before, between, after = get_words_in_sentence(id_e1, id_e2, entities, analysis)
    shared_verb_keyword = shared_verb_keyword_search(id_e1, id_e2, entities, entsToNodes, analysis, graph, clues_before,
                                                     clues_between, clues_after)
    verb_keyword = path_keyword_search(id_e1, id_e2, "VB", entities, entsToNodes, analysis, graph, clues_before,
                                       clues_between, clues_after)

    noun_keyword = path_keyword_search(id_e1, id_e2, "NN", entities, entsToNodes, analysis, graph, clues_before,
                                       clues_between, clues_after)
    adj_keyword = path_keyword_search(id_e1, id_e2, "JJ", entities, entsToNodes, analysis, graph, clues_before,
                                      clues_between, clues_after)
    cc_keyword = path_keyword_search(id_e1, id_e2, "CC", entities, entsToNodes, analysis, graph, clues_before,
                                     clues_between, clues_after)
    any_keyword = path_keyword_search(id_e1, id_e2, "any", entities, entsToNodes, analysis, graph, clues_before,
                                      clues_between, clues_after)

    sentence_keyword = whole_sentence_keyword_search(id_e1, id_e2, entities, analysis, clues_before, clues_between,
                                                     clues_after)

    # ----------------------------------------------------------------------------------------------
    # PHRASE PROPERTIES TESTS
    # ----------------------------------------------------------------------------------------------

    same_verb = entities_under_same_verb(id_e1, id_e2, entsToNodes, analysis, graph)
    rels = find_dependencies_between(id_e1, id_e2, entities, entsToNodes, analysis, graph)
    _2under1 = second_entity_under_first(id_e1, id_e2, entsToNodes, graph)
    rel_type = check_dependencies_between(id_e1, id_e2, entities, entsToNodes, analysis, graph)

    # ----------------------------------------------------------------------------------------------
    # PHRASE SEARCH TESTS
    # ----------------------------------------------------------------------------------------------

    preposition = preposition_in_phrase(id_e1, id_e2, entities, entsToNodes, analysis, graph)
    negation = negation_in_phrase(id_e1, id_e2, entsToNodes, analysis, graph)
    negative_conj = negative_conjunction_between_entities(id_e1, id_e2, entities, entsToNodes, analysis, graph)

    # First Step: identify whether or not a potential interaction exists.
    # Properties to check:
    # if the entities have a preposition in their shared phrase

    if preposition:

        # Second Step: weed out false positive.
        # False positives: identified by checking for lack of preposition,
        # or presence of negative conjunction between them

        if negative_conj:
            return 0, 'null'

        # Third step: identify type of interaction.
        # checks the relations between the entities for known
        # relations for each ddi type. Also checks the verb that both entities share
        # for keywords, and returns the associated ddi type for that keyword.
        # Prioritises the shared_verb keyword over the rel_type, if both are not null.
        if rel_type == shared_verb_keyword and rel_type:
            ddi_type = rel_type
        elif rel_type and not shared_verb_keyword:
            ddi_type = rel_type
        elif shared_verb_keyword and not rel_type:
            ddi_type = shared_verb_keyword
        elif rel_type and shared_verb_keyword:
            ddi_type = shared_verb_keyword

        if ddi_type:
            return 1, ddi_type
        else:
            return 0, 'null'
    else:
        return 0, 'null'
