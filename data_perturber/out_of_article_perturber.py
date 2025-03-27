import penman
import random
from typing import Tuple
from data_perturber.utils import penman_to_networkx, networkx_to_penman
from faker import Faker

# Initialize Faker
fake = Faker(['id_ID', 'en_US'])  # Support both Indonesian and English

def insertOutOfArticleError(graph: penman.Graph) -> Tuple[penman.Graph, dict]:
    """Insert out-of-article errors by adding extraneous information.
    
    Args:
        graph: Input AMR graph
        
    Returns:
        Tuple of (perturbed graph, changelog)
    """
    # Create a simplified approach that modifies the triple directly
    triples = graph.triples.copy()
    variables = graph.variables()
    
    # Generate dynamic extraneous concepts to add
    person_name = fake.name()
    company_name = fake.company()
    location_name = fake.city()
    year = str(fake.year())
    
    extraneous_concepts = [
        ('person', ':name', f'"{person_name}"'),
        ('company', ':name', f'"{company_name}"'),
        ('location', ':name', f'"{location_name}"'),
        ('date-entity', ':year', f'"{year}"')
    ]
    
    # Select random concept to add
    concept, rel, value = random.choice(extraneous_concepts)
    
    # Generate unique variable IDs that don't exist in the graph
    max_var_num = 0
    for var in variables:
        if var.startswith('z') and var[1:].isdigit():
            max_var_num = max(max_var_num, int(var[1:]))
    
    new_node_var = f"z{max_var_num + 1}"
    name_node_var = f"z{max_var_num + 2}" if rel == ':name' else None
    
    # Find a suitable existing node to attach to
    existing_nodes = [var for var in variables]
    if existing_nodes:
        connect_node = random.choice(existing_nodes)
        
        # Add the new concept node
        triples.append((new_node_var, ':instance', concept))
        
        # Connect to existing node
        triples.append((connect_node, ':mod', new_node_var))
        
        if rel == ':name':
            # Add name node
            triples.append((name_node_var, ':instance', 'name'))
            triples.append((new_node_var, ':name', name_node_var))
            
            # Add value (op1)
            op_value = value.strip('"')  # Remove quotes
            triples.append((name_node_var, ':op1', f'"{op_value}"'))
        else:
            # Add attribute
            triples.append((new_node_var, rel.lstrip(':'), value))
    
    changelog = {
        'type': 'out_of_article_error',
        'added_concept': concept,
        'added_relation': rel,
        'added_value': value,
        'connected_to': connect_node if existing_nodes else None
    }
    
    # Create a new graph with the modified triples
    new_graph = penman.Graph(triples)
    new_graph.metadata = graph.metadata
    new_graph.epidata = graph.epidata.copy()
    
    return new_graph, changelog
