import penman
import random
from typing import Tuple
from data_perturber.utils import penman_to_networkx, networkx_to_penman

def insertOutOfArticleError(graph: penman.Graph) -> Tuple[penman.Graph, dict]:
    """Insert out-of-article errors by adding extraneous information.
    
    Args:
        graph: Input AMR graph
        
    Returns:
        Tuple of (perturbed graph, changelog)
    """
    G = penman_to_networkx(penman_graph=graph)
    changelog = {}
    
    # Common extraneous concepts to add
    extraneous_concepts = [
        ('person', ':name', 'Anonymous'),
        ('company', ':name', 'CompanyX'),
        ('location', ':name', 'Somewhere'),
        ('date-entity', ':year', '2025')
    ]
    
    # Select random concept to add
    concept, rel, value = random.choice(extraneous_concepts)
    
    # Generate unique node ID
    new_node = f"x{random.randint(1000,9999)}"
    name_node = None
    
    # Add new nodes and edges
    G.add_node(new_node, label=concept)
    if rel == ':name':
        name_node = f"n{random.randint(1000,9999)}"
        G.add_node(name_node, label='name')
        G.add_edge(new_node, name_node, key=rel)
        G.add_edge(name_node, value, key=':op1')
    else:
        G.add_edge(new_node, value, key=rel)
    
    # Connect to random existing node to prevent disconnected graph
    existing_nodes = [n for n in G.nodes() if n not in [new_node, name_node] if n != value]
    if existing_nodes:
        random_node = random.choice(existing_nodes)
        G.add_edge(random_node, new_node, key=':mod')
    
    changelog = {
        'type': 'out_of_article_error',
        'added_concept': concept,
        'added_relation': rel,
        'added_value': value,
        'connected_to': random_node if existing_nodes else None
    }
    
    return networkx_to_penman(G), changelog
