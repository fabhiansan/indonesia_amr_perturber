import penman
import random
from typing import Tuple
from data_perturber.utils import penman_to_networkx, networkx_to_penman

def insertDiscourseError(graph: penman.Graph) -> Tuple[penman.Graph, dict]:
    """Insert discourse link errors by modifying temporal/causal relations.
    
    Args:
        graph: Input AMR graph
        
    Returns:
        Tuple of (perturbed graph, changelog)
    """
    G = penman_to_networkx(graph)
    changelog = {}
    
    # Find temporal/causal relations to modify
    temporal_relations = [':time', ':duration', ':before', ':after']
    causal_relations = [':cause', ':condition', ':purpose']
    
    # Get edges that could be modified
    edges = [(u,v,d) for u,v,d in G.edges(data=True) 
             if d['label'] in temporal_relations + causal_relations]
    
    if not edges:
        return graph, {'error': 'No discourse relations found'}
    
    # Select random edge to modify
    u, v, d = random.choice(edges)
    old_rel = d['label']
    
    # Modify relation based on type
    if old_rel in temporal_relations:
        # Swap temporal relations
        new_rel = random.choice([r for r in temporal_relations if r != old_rel])
    else:
        # Swap causal relations
        new_rel = random.choice([r for r in causal_relations if r != old_rel])
    
    # Update edge
    G.edges[u, v]['label'] = new_rel
    changelog = {
        'type': 'discourse_error',
        'old_relation': old_rel,
        'new_relation': new_rel,
        'nodes': (u, v)
    }
    
    return networkx_to_penman(G), changelog
