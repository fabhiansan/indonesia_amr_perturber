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
    changelog = []
    
    # Get all predicate nodes that can be manipulated
    predicate_nodes = []
    for node in nx_graph.nodes():
        # Check if this is a predicate node (has an instance edge with a label ending in "-01", "-02", etc.)
        for _, instance_value, edge_data in nx_graph.out_edges(node, data=True):
            if edge_data.get('label') == ':instance' and isinstance(instance_value, str):
                if "-" in instance_value and len(instance_value.split("-")) > 1:
                    if instance_value.split("-")[-1].isdigit():
                        predicate_nodes.append((node, instance_value))
                        break
    
    # If no predicate nodes found, try to add polarity to the root node
    if not predicate_nodes and len(list(nx_graph.nodes())) > 0:
        root_candidates = [node for node in nx_graph.nodes() if nx_graph.in_degree(node) == 0]
        if root_candidates:
            root_node = root_candidates[0]
            # Add polarity to root node
            has_polarity = False
            for _, target, edge_data in nx_graph.out_edges(root_node, data=True):
                if edge_data.get('label') == ':polarity':
                    has_polarity = True
                    break
            
            if not has_polarity:
                nx_graph.add_edge(root_node, '-', label=':polarity')
                changelog.append({f"Added polarity to {root_node}": "negative"})
            else:
                # Remove existing polarity
                polarity_targets = []
                for _, target, edge_data in nx_graph.out_edges(root_node, data=True):
                    if edge_data.get('label') == ':polarity':
                        polarity_targets.append(target)
                
                for target in polarity_targets:
                    nx_graph.remove_edge(root_node, target)
                changelog.append({f"Removed polarity from {root_node}": "positive"})
            
            return networkx_to_penman(nx_graph), changelog
    
    # Randomly select up to n_wrong predicates to modify
    if predicate_nodes:
        selected_nodes = random.sample(predicate_nodes, min(n_wrong, len(predicate_nodes)))
        
        for node, instance_value in selected_nodes:
            predicate_base = instance_value.split("-")[0]
            antonyms = get_indonesian_antonyms(predicate_base)
            
            if antonyms:
                # Use antonym substitution
                selected_antonym = random.choice(antonyms)
                
                # Handle "tidak" and "tak" prefixes
                if selected_antonym.startswith('tidak ') or selected_antonym.startswith('tak '):
                    base_word = selected_antonym.replace('tidak ', '').replace('tak ', '')
                    
                    # Find the original instance value
                    instance_edges = []
                    for _, target, edge_data in nx_graph.out_edges(node, data=True):
                        if edge_data.get('label') == ':instance':
                            instance_edges.append((node, target, edge_data))
                    
                    # Update the instance value
                    if instance_edges:
                        nx_graph.remove_edge(node, instance_edges[0][1])
                        suffix = instance_value.split("-")[-1] if "-" in instance_value else "01"
                        new_instance = f"{base_word}-{suffix}"
                        nx_graph.add_edge(node, new_instance, label=':instance')
                    
                    # Add polarity if it doesn't exist
                    has_polarity = False
                    for _, target, edge_data in nx_graph.out_edges(node, data=True):
                        if edge_data.get('label') == ':polarity':
                            has_polarity = True
                            break
                    
                    if not has_polarity:
                        nx_graph.add_edge(node, '-', label=':polarity')
                else:
                    # Simple antonym replacement
                    instance_edges = []
                    for _, target, edge_data in nx_graph.out_edges(node, data=True):
                        if edge_data.get('label') == ':instance':
                            instance_edges.append((node, target, edge_data))
                    
                    if instance_edges:
                        nx_graph.remove_edge(node, instance_edges[0][1])
                        suffix = instance_value.split("-")[-1] if "-" in instance_value else "01"
                        new_instance = f"{selected_antonym}-{suffix}"
                        nx_graph.add_edge(node, new_instance, label=':instance')
                
                changelog.append({instance_value: selected_antonym})
            else:
                # Toggle polarity if no antonyms found
                has_polarity = False
                polarity_target = None
                
                for _, target, edge_data in nx_graph.out_edges(node, data=True):
                    if edge_data.get('label') == ':polarity':
                        has_polarity = True
                        polarity_target = target
                        break
                
                if has_polarity:
                    # Remove polarity
                    nx_graph.remove_edge(node, polarity_target)
                    changelog.append({f"Removed polarity from {instance_value}": "positive"})
                else:
                    # Add negative polarity
                    nx_graph.add_edge(node, '-', label=':polarity')
                    changelog.append({f"Added polarity to {instance_value}": "negative"})
    
    return networkx_to_penman(nx_graph), changelog
