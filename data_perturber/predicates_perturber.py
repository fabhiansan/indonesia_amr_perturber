import penman
from penman import Graph
import random
import networkx as nx
from .utils import penman_to_networkx, networkx_to_penman, get_indonesian_antonyms


def insertWrongPredicates(amr_graph: Graph, n_wrong=1):
    """
    Predicate Error. Predicate errors occur when
    the predicate in a summary does not align with
    the information in the source document. We simu-
    late this type of error based on two processes: (1)
    by adding or removing polarity and (2) through
    the substitution of a predicate with its antonym.
    By directly adding or removing polarity to the
    concpets, we change the negation in the sen-
    tence. Another approach is the antonym substi-
    tution. Here, we replace the concepts with their
    antonyms that holds Antonym, NotDesires, NotCa-
    pableOf, and NotHasProperty relations in Concept-
    Net (Speer and Havasi, 2012), and therefore modify
    the sentence-level relations.
    
    Insert n_wrong wrong predicted nodes into an AMR graph.
    
    Args:
        amr_graph (str): AMR graph string
        n_wrong (int): Number of wrong predicted nodes to insert
        
    Returns:
        str: AMR graph string with wrong predicted nodes inserted
    
    """
    nx_graph = penman_to_networkx(amr_graph)
    copy_nx_graph = nx_graph.copy()
    changelog = []
    all_new_graphs = []

    # manipulation
    for i in copy_nx_graph.nodes:
        if "-" in i and len(i.split("-")[1]) == 2:
            preds = i.split("-")[0]
            
            ants = get_indonesian_antonyms(preds)
            
            if len(ants) > 0:
                selected_preds = random.choice(ants)
                original_pred = selected_preds
                
                # Handle "tidak" and "tak" prefixes with proper AMR polarity
                if selected_preds.startswith('tidak '):
                    base_word = selected_preds.replace('tidak ', '')
                    new_graph = nx_graph.copy()
                    new_graph.add_node('-', label='-')
                    new_graph.add_edge(i, '-', label=':polarity')
                    new_graph = nx.relabel_nodes(new_graph, {i: base_word})
                elif selected_preds.startswith('tak '):
                    base_word = selected_preds.replace('tak ', '')
                    new_graph = nx_graph.copy()
                    new_graph.add_node('-', label='-')
                    new_graph.add_edge(i, '-', label=':polarity')
                    new_graph = nx.relabel_nodes(new_graph, {i: base_word})
                else:
                    new_graph = nx.relabel_nodes(nx_graph, {i : selected_preds})
                changelog.append({i : original_pred})
                all_new_graphs.append(networkx_to_penman(new_graph))
            else:
                # todo masih overwrite nx_graph
                pair = [edge for edge in nx_graph.edges if i in edge]
                zIndex = pair[0][0]
                nx_graph.add_node("-")
                nx_graph.add_edge(zIndex, "-", label=":polarity")
                
                # todo - remove polarity if exist
                
                
                
    if not all_new_graphs:
        # If no antonyms found, return original graph with empty changelog
        return amr_graph, []
    return all_new_graphs[0], changelog
