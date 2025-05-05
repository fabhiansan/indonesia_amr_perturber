import penman
import random
from typing import Tuple
from data_perturber.utils import penman_to_networkx, networkx_to_penman

def insertDiscourseError(graph: penman.Graph) -> Tuple[penman.Graph, dict]:
    """Insert discourse link errors by modifying temporal/causal relations or arguments.

    Args:
        graph: Input AMR graph

    Returns:
        Tuple of (perturbed graph, changelog)
    """
    G = penman_to_networkx(graph)
    changelog = {}
    original_top = graph.top # Get the original top node

    # Define possible discourse error types
    error_types = ['swap_temporal', 'swap_causal', 'modify_causal_args'] # Add more types later

    # Select a random error type
    selected_error_type = random.choice(error_types)

    if selected_error_type == 'swap_temporal':
        # Use Indonesian temporal relations
        temporal_relations = [':time', ':durasi', ':sebelum', ':setelah']
        # Find edges with either English or Indonesian temporal relations for modification
        english_temporal_relations = [':time', ':duration', ':before', ':after'] # Keep English for finding existing
        edges = [(u,v,d) for u,v,d in G.edges(data=True)
                 if d['label'] in english_temporal_relations] # Find existing English ones

        if not edges:
            # If no English temporal relations found, try finding Indonesian ones if they might exist from previous runs
            edges = [(u,v,d) for u,v,d in G.edges(data=True)
                     if d['label'] in temporal_relations]
            if not edges:
                 return graph, {'error': 'No temporal relations (English or Indonesian) found for swapping'}


        u, v, d = random.choice(edges)
        old_rel = d['label']

        # Ensure the new relation is one of the Indonesian/standard temporal relations
        new_rel = random.choice([r for r in temporal_relations if r != old_rel])

        G.edges[u, v]['label'] = new_rel
        changelog = {
            'type': 'discourse_error',
            'subtype': 'swap_temporal',
            'old_relation': old_rel,
            'new_relation': new_rel,
            'nodes': (u, v)
        }

    elif selected_error_type == 'swap_causal':
        causal_relations = [':cause', ':condition', ':purpose']
        edges = [(u,v,d) for u,v,d in G.edges(data=True)
                 if d['label'] in causal_relations]

        if not edges:
            return graph, {'error': 'No causal relations found for swapping'}

        u, v, d = random.choice(edges)
        old_rel = d['label']
        new_rel = random.choice([r for r in causal_relations if r != old_rel])

        G.edges[u, v]['label'] = new_rel
        changelog = {
            'type': 'discourse_error',
            'subtype': 'swap_causal',
            'old_relation': old_rel,
            'new_relation': new_rel,
            'nodes': (u, v)
        }

    elif selected_error_type == 'modify_causal_args':
        # Find nodes that are instances of 'cause-01'
        cause_nodes = [
            node for node in G.nodes()
            if any(d.get('label') == ':instance' and v == 'cause-01' for u, v, d in G.out_edges(node, data=True))
        ]

        if not cause_nodes:
            return graph, {'error': 'No cause-01 nodes found for argument modification'}

        # Select a random cause node
        cause_node = random.choice(cause_nodes)

        # Find ARG0 and ARG1 edges connected to this cause node
        arg_edges = [
            (u, v, d) for u, v, d in G.out_edges(cause_node, data=True)
            if d.get('label') in [':ARG0', ':ARG1']
        ]

        if len(arg_edges) < 2:
             return graph, {'error': f'Cause node {cause_node} does not have both :ARG0 and :ARG1 for swapping'}

        # Randomly choose to swap ARG0/ARG1 or change one of them
        modify_action = random.choice(['swap_args', 'change_arg_label'])

        if modify_action == 'swap_args':
            arg0_edge = next((e for e in arg_edges if e[2]['label'] == ':ARG0'), None)
            arg1_edge = next((e for e in arg_edges if e[2]['label'] == ':ARG1'), None)

            if arg0_edge and arg1_edge:
                # Remove old edges
                G.remove_edge(arg0_edge[0], arg0_edge[1])
                G.remove_edge(arg1_edge[0], arg1_edge[1])

                # Add new edges with swapped labels
                G.add_edge(cause_node, arg0_edge[1], label=':ARG1')
                G.add_edge(cause_node, arg1_edge[1], label=':ARG0')

                changelog = {
                    'type': 'discourse_error',
                    'subtype': 'swap_causal_args',
                    'cause_node': cause_node,
                    'description': f'Swapped :ARG0 and :ARG1 for {cause_node}'
                }
            else:
                 return graph, {'error': f'Could not find both :ARG0 and :ARG1 edges for {cause_node} to swap'}

        elif modify_action == 'change_arg_label':
            # Select one of the ARG0/ARG1 edges
            edge_to_modify = random.choice(arg_edges)
            old_label = edge_to_modify[2]['label']
            new_label = ':ARG1' if old_label == ':ARG0' else ':ARG0' # Swap label

            # Update the edge label
            G.edges[edge_to_modify[0], edge_to_modify[1]]['label'] = new_label

            changelog = {
                'type': 'discourse_error',
                'subtype': 'change_causal_arg_label',
                'cause_node': cause_node,
                'old_relation': old_label,
                'new_relation': new_label,
                'nodes': (edge_to_modify[0], edge_to_modify[1])
            }


    else:
        # Should not happen with current error_types list
        return graph, {'error': f'Unknown error type selected: {selected_error_type}'}


    # Convert back to Penman graph
    try:
    # Convert back to Penman graph, preserving the original top node
        result_graph = networkx_to_penman(G, top=original_top) # Pass original_top
        # Basic validation
        if result_graph.top is None or not result_graph.triples:
             raise ValueError("possibly disconnected graph or empty graph after perturbation")
        return result_graph, changelog
    except Exception as e:
        # If conversion fails, return the original graph and log error
        print(f"Error converting modified graph to Penman: {str(e)}")
        return graph, {'error': f'Graph conversion failed: {str(e)}', 'original_changelog': changelog}
