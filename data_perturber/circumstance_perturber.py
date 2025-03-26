import penman
from penman import Graph
import random
import networkx as nx
from data_perturber.utils import penman_to_networkx, networkx_to_penman, get_indonesian_antonyms

def insertCircumstanceError(amr_graph: Graph, error_type="both"):
    """
    Circumstance Error. Circumstance errors in summaries emerge when there is incorrect 
    or misleading information regarding the context of predicate interactions, specifically 
    in terms of location, time, and modality.
    
    These errors are mainly created in two ways:
    1. By intensifying the modality, which alters the degree of certainty or possibility
       expressed in the statement
    2. By substituting specific entities like locations and times
    
    Args:
        amr_graph (Graph): AMR graph 
        error_type (str): Type of error to insert - "modality", "entity", or "both"
        
    Returns:
        tuple: (List of perturbed graphs, changelog)
    """
    perturbed_graphs = []
    changelog = []
    
    # Convert to networkx for manipulation
    nx_graph = penman_to_networkx(amr_graph)
    
    if error_type in ["modality", "both"]:
        modality_graphs, modality_changes = modify_modality(nx_graph.copy())
        perturbed_graphs.extend(modality_graphs)
        changelog.extend(modality_changes)
    
    if error_type in ["entity", "both"]:
        entity_graphs, entity_changes = substitute_circumstance_entities(nx_graph.copy())
        perturbed_graphs.extend(entity_graphs)
        changelog.extend(entity_changes)
    
    if not perturbed_graphs:
        return amr_graph, []
    return perturbed_graphs[0], changelog

def modify_modality(nx_graph):
    """
    Modify modality in AMR graph by replacing modality concepts with stronger/weaker ones.
    
    Args:
        nx_graph: NetworkX graph representation of AMR
        
    Returns:
        tuple: (List of perturbed graphs, changelog)
    """
    perturbed_graphs = []
    changelog = []
    
    # Common modality concepts and their intensified versions
    # Format: {original: intensified}
    modality_mapping = {
        "possible": "obligate",
        "maybe": "certain",
        "doubt": "believe",
        "think": "know",
        "permit": "obligate",
        "might": "must",
        "may": "will",
        "could": "should",
        "suggest": "demand",
        "ask": "command",
        "request": "require",
    }
    
    # Find all nodes that may represent modality
    for node in list(nx_graph.nodes()):
        # Check if node is a string value (concept) that exists in our mapping
        if isinstance(node, str) and any(modal in node.lower() for modal in modality_mapping.keys()):
            # Get the base concept (without instance number)
            if "-" in node:
                base_concept = node.split("-")[0].lower()
            else:
                base_concept = node.lower()
            
            # Find matching concepts to replace
            for original, intensified in modality_mapping.items():
                if original in base_concept:
                    # Create a new graph with the replaced modality
                    if "-" in node:
                        # Preserve the instance number
                        instance_num = node.split("-")[1]
                        new_node = f"{intensified}-{instance_num}"
                    else:
                        new_node = intensified
                    
                    # Create a copy to avoid modifying the original
                    graph_copy = nx_graph.copy()
                    
                    # Relabel the node
                    graph_copy = nx.relabel_nodes(graph_copy, {node: new_node})
                    
                    # Convert back to penman
                    penman_graph = networkx_to_penman(graph_copy)
                    
                    perturbed_graphs.append(penman_graph)
                    changelog.append({f"Changed modality": f"'{node}' → '{new_node}'"})
                    break  # Only apply one replacement per node
    
    return perturbed_graphs, changelog

def substitute_circumstance_entities(nx_graph):
    """
    Substitute circumstance entities in AMR graph by identifying and replacing
    nodes related to time, location, and other circumstantial elements.
    
    Args:
        nx_graph: NetworkX graph representation of AMR
        
    Returns:
        tuple: (List of perturbed graphs, changelog)
    """
    perturbed_graphs = []
    changelog = []
    
    # Circumstance relations in AMR
    circumstance_relations = [
        ":time", ":time-of", ":duration", ":decade", ":year", ":month", ":day", 
        ":weekday", ":dayperiod", ":season", ":timezone", 
        ":location", ":source", ":destination", ":path", ":direction",
        ":manner", ":purpose", ":cause", ":condition", ":concession"
    ]
    
    # Find all edges with circumstance relations
    circumstance_edges = []
    for u, v, data in nx_graph.edges(data=True):
        if data.get('label') in circumstance_relations:
            circumstance_edges.append((u, v, data))
    
    if not circumstance_edges:
        return perturbed_graphs, changelog  # No circumstance entities to modify
    
    # Randomly select one circumstance edge to modify
    if circumstance_edges:
        edge_to_modify = random.choice(circumstance_edges)
        source, target, data = edge_to_modify
        relation = data.get('label')
        
        # Create a modified graph copy
        graph_copy = nx_graph.copy()
        
        # Strategy 1: Replace the entity with a different but plausible entity
        if isinstance(target, str) and not target.startswith(":"):
            # For named entities, we could change them to different names
            if "-" in target:
                base_entity = target.split("-")[0]
                # Get an "antonym" or different entity
                replacements = get_entity_replacements(base_entity)
                if replacements:
                    new_entity = random.choice(replacements)
                    # Preserve instance number
                    instance_num = target.split("-")[1]
                    new_target = f"{new_entity}-{instance_num}"
                    
                    # Relabel the node
                    graph_copy = nx.relabel_nodes(graph_copy, {target: new_target})
                    
                    # Convert back to penman
                    penman_graph = networkx_to_penman(graph_copy)
                    
                    perturbed_graphs.append(penman_graph)
                    changelog.append({f"Changed {relation[1:]} entity": f"'{target}' → '{new_target}'"})
        
        # Strategy 2: Invert time relationships or change location specificity
        # For example, change :time to :time-of, or :location to :destination
        if not perturbed_graphs:  # If first strategy didn't work
            if relation in [":time", ":time-of"]:
                new_relation = ":time" if relation == ":time-of" else ":time-of"
            elif relation in [":location", ":source", ":destination"]:
                other_relations = [r for r in [":location", ":source", ":destination"] if r != relation]
                new_relation = random.choice(other_relations)
            else:
                return perturbed_graphs, changelog  # Skip if no suitable substitution
            
            # Create a copy and modify the edge
            graph_copy = nx_graph.copy()
            graph_copy.remove_edge(source, target)
            graph_copy.add_edge(source, target, label=new_relation)
            
            # Convert back to penman
            penman_graph = networkx_to_penman(graph_copy)
            
            perturbed_graphs.append(penman_graph)
            changelog.append({f"Changed relation": f"'{relation}' → '{new_relation}'"})
    
    return perturbed_graphs, changelog

def get_entity_replacements(entity, n=3):
    """
    Get replacement entities based on concept type.
    For simplicity, we use some predefined replacements for common categories.
    
    Args:
        entity (str): Original entity
        n (int): Number of replacements to generate
        
    Returns:
        list: Possible replacement entities
    """
    # Simple dictionary of entity categories and examples
    entity_categories = {
        # Locations
        "city": ["Jakarta", "Surabaya", "Bandung", "Medan", "Makassar", "Semarang"],
        "country": ["Indonesia", "Malaysia", "Singapore", "Australia", "Japan", "Korea"],
        # Times
        "month": ["January", "February", "March", "April", "May", "June", "July"],
        "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "year": ["2021", "2022", "2023", "2024", "2025"],
        # Generic replacements
        "location": ["selatan", "utara", "barat", "timur", "pusat", "pinggir"],
        "time": ["pagi", "siang", "sore", "malam", "dini hari"],
    }
    
    # Try to identify entity category
    for category, examples in entity_categories.items():
        if entity.lower() in [ex.lower() for ex in examples]:
            # Return other examples from same category
            return [ex for ex in examples if ex.lower() != entity.lower()]
    
    # If no category match, fall back to using antonyms
    antonyms = get_indonesian_antonyms(entity, max_antonyms=n)
    if antonyms:
        return antonyms
    
    # Last resort - generate placeholder replacements
    return [f"alternate_{entity}_{i}" for i in range(1, n+1)]
