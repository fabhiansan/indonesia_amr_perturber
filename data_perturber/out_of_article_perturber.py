import penman
import random
from typing import Tuple, Dict, Any, List, Optional
from penman import Graph
import networkx as nx
from data_perturber.utils import penman_to_networkx, networkx_to_penman
from faker import Faker

# Initialize Faker
fake = Faker('id_ID')  # Prioritize Indonesian locale for consistency

# Indonesian month names mapping
INDONESIAN_MONTHS = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April", 
    5: "Mei",
    6: "Juni",
    7: "Juli",
    8: "Agustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember"
}

def insertOutOfArticleError(graph: Graph) -> Tuple[Graph, Dict[str, Any]]:
    """Insert out-of-article errors by adding extraneous information.
    
    This function adds information not present in the original document by
    attaching new nodes with generated content to existing nodes in the graph.
    The new information includes persons, locations, dates, organizations, and more,
    with a focus on Indonesian-specific content.
    
    Args:
        graph: Input AMR graph
        
    Returns:
        Tuple containing:
            - The perturbed AMR graph
            - A changelog dictionary with details of the changes made
    """
    try:
        # Create a copy of the input graph to avoid modifying the original
        graph_copy = penman.decode(penman.encode(graph))
        triples = graph_copy.triples.copy()
        variables = list(graph_copy.variables())  # Convert to list to make it subscriptable
        
        # Generate dynamic extraneous concepts to add
        extraneous_concepts = _generate_extraneous_concepts()
        
        # Select random concept to add
        concept_data = random.choice(extraneous_concepts)
        concept_type = concept_data['type']
        concept_value = concept_data['value']
        relation = random.choice(concept_data['relations'])
        
        # Find nodes in the graph that can accept this type of extraneous information
        suitable_nodes = []
        for var in variables:
            # Check if this node is suitable for our concept type
            for triple in triples:
                if triple[0] == var and triple[1] == ':instance':
                    instance = triple[2]
                    # Simple heuristic to find suitable nodes
                    if concept_type == 'date-entity' and any(t in instance for t in ['time', 'date', 'tanggal', 'waktu']):
                        suitable_nodes.append(var)
                    elif concept_type == 'person' and any(t in instance for t in ['person', 'orang', 'pria', 'wanita']):
                        suitable_nodes.append(var)
                    elif concept_type == 'location' and any(t in instance for t in ['location', 'tempat', 'lokasi']):
                        suitable_nodes.append(var)
                    elif concept_type == 'organization' and any(t in instance for t in ['organization', 'perusahaan', 'lembaga']):
                        suitable_nodes.append(var)
        
        # If no suitable node is found, add to any node
        if not suitable_nodes:
            suitable_nodes = variables
        
        # Select a random node to attach to
        if suitable_nodes:
            target_node = random.choice(suitable_nodes)
            
            # Generate unique variable IDs that don't exist in the graph
            max_var_num = 0
            for var in variables:
                if var.startswith('z') and var[1:].isdigit():
                    max_var_num = max(max_var_num, int(var[1:]))
            
            # Create new nodes with proper instance relationships
            new_node_var = f"z{max_var_num + 1}"
            
            # Add the new concept node with proper instance
            triples.append((new_node_var, ':instance', concept_type))
            
            # Connect to existing node with appropriate relation
            triples.append((target_node, relation, new_node_var))
            
            # Depending on the concept type, add appropriate attributes
            if concept_type in ['person', 'organization', 'location']:
                # Add a name node
                name_node_var = f"z{max_var_num + 2}"
                triples.append((name_node_var, ':instance', 'name'))
                triples.append((new_node_var, ':name', name_node_var))
                
                # Split into name parts
                name_parts = concept_value.split()
                for i, part in enumerate(name_parts[:3]):  # Limit to 3 name parts
                    triples.append((name_node_var, f':op{i+1}', f'"{part}"'))
                    
            elif concept_type == 'date-entity':
                if 'tahun' in concept_value:
                    year_match = concept_value.split('tahun ')[1].strip() if 'tahun ' in concept_value else ''
                    if year_match.isdigit():
                        triples.append((new_node_var, ':year', f'"{year_match}"'))
                elif any(month in concept_value.lower() for month in ['januari', 'februari', 'maret', 'april', 'mei', 'juni', 'juli', 'agustus', 'september', 'oktober', 'november', 'desember']):
                    triples.append((new_node_var, ':month', f'"{concept_value}"'))
                else:
                    triples.append((new_node_var, ':time', f'"{concept_value}"'))
                    
            elif concept_type == 'quantity':
                triples.append((new_node_var, ':quant', f'"{concept_value}"'))
                
            else:
                # For other types, add a generic value
                triples.append((new_node_var, ':value', f'"{concept_value}"'))
            
            # Create a new graph with the modified triples
            new_graph = penman.Graph(triples, top=graph_copy.top)
            
            # Create changelog
            changelog = {
                'type': 'out_of_article_error',
                'added_concept': concept_type,
                'added_relation': relation,
                'added_value': concept_value,
                'connected_to': target_node
            }
            
            return new_graph, changelog
        
        # Return original graph and error if no nodes available
        return graph, {'error': 'No suitable nodes found in graph'}
    
    except Exception as e:
        # Provide fallback in case of error
        return graph, {'error': f'Failed to insert out-of-article error: {str(e)}'}

def _generate_extraneous_concepts() -> List[Dict[str, Any]]:
    """
    Generate a list of extraneous concepts to potentially add to the AMR graph.
    Focuses on Indonesian-specific content.
    
    Returns:
        A list of dictionaries with concept types and values
    """
    concepts = []
    
    # Person entities (Indonesian names)
    concepts.append({
        'type': 'person',
        'value': fake.name(),
        'relations': [':ARG0-of', ':ARG1-of', ':mod', ':poss', ':topic']
    })
    
    # Organization entities (Indonesian companies and institutions)
    company_formats = [
        f"PT {fake.company()}",
        f"{fake.company()} (Persero) Tbk",
        f"Yayasan {fake.last_name()}",
        f"Universitas {fake.city()}",
        f"Bank {fake.last_name()} Indonesia",
        f"Rumah Sakit {fake.last_name()}"
    ]
    concepts.append({
        'type': 'organization',
        'value': random.choice(company_formats),
        'relations': [':ARG0-of', ':ARG1-of', ':mod', ':org', ':poss']
    })
    
    # Location entities (Indonesian locations)
    location_formats = [
        fake.city(),
        fake.state(),
        fake.street_name(),
        f"Kabupaten {fake.city()}",
        f"Kecamatan {fake.city()}",
        f"Desa {fake.city()}"
    ]
    concepts.append({
        'type': 'location',
        'value': random.choice(location_formats),
        'relations': [':location', ':source', ':destination', ':path', ':mod']
    })
    
    # Time entities (dates with Indonesian format)
    date_formats = []
    
    # Get a random date
    random_date = fake.date_this_decade()
    # Format with Indonesian month names
    formatted_date = f"{random_date.day} {INDONESIAN_MONTHS[random_date.month]} {random_date.year}"
    weekday = fake.day_of_week()
    
    date_formats = [
        formatted_date,
        f"tahun {fake.year()}",
        f"bulan {INDONESIAN_MONTHS[random.randint(1, 12)]}",
        f"hari {weekday}",
        f"{weekday}, {formatted_date}"
    ]
    concepts.append({
        'type': 'date-entity',
        'value': random.choice(date_formats),
        'relations': [':time', ':year', ':month', ':day', ':mod']
    })
    
    # Quantity entities
    concepts.append({
        'type': 'quantity',
        'value': str(random.randint(100, 9999)),
        'relations': [':quant', ':value', ':mod', ':unit']
    })
    
    # Event entities (Indonesian events)
    event_formats = [
        f"Festival {fake.word().capitalize()}",
        f"Konferensi {fake.word().capitalize()} Nasional",
        f"Pameran {fake.word().capitalize()} Indonesia",
        f"Perayaan {fake.word().capitalize()}",
        f"Pemilihan Umum {fake.year()}"
    ]
    concepts.append({
        'type': 'event',
        'value': random.choice(event_formats),
        'relations': [':ARG1-of', ':topic', ':mod', ':time']
    })
    
    return concepts

def _find_suitable_target_nodes(nx_graph: nx.DiGraph, concept_type: str) -> List[str]:
    """
    Find nodes in the graph that would make suitable targets for the given concept type.
    
    Args:
        nx_graph: The NetworkX graph
        concept_type: The type of concept to add
        
    Returns:
        A list of node IDs that are suitable for attaching the concept
    """
    suitable_nodes = []
    
    # Define concept type to target node instance mappings
    concept_target_mappings = {
        'person': ['person', 'orang', 'pria', 'wanita', 'manusia', 'pejabat', 'pekerja'],
        'organization': ['perusahaan', 'organisasi', 'lembaga', 'institusi', 'kelompok'],
        'location': ['tempat', 'lokasi', 'negara', 'kota', 'wilayah', 'area', 'daerah'],
        'date-entity': ['waktu', 'date-entity', 'tanggal', 'hari', 'bulan', 'tahun', 'era'],
        'quantity': ['have-quantity', 'jumlah', 'nilai', 'angka', 'rate', 'harga'],
        'event': ['acara', 'kegiatan', 'kejadian', 'peristiwa', 'event', 'situasi']
    }
    
    # Look for instance nodes that match our target types
    for node, data in nx_graph.nodes(data=True):
        instance = data.get('instance')
        if instance:
            # Check if this node instance would be suitable for our concept type
            target_instances = concept_target_mappings.get(concept_type, [])
            if any(target in instance.lower() for target in target_instances):
                suitable_nodes.append(node)
            
            # Also consider root nodes as they often represent main concepts
            if nx_graph.in_degree(node) == 0:
                suitable_nodes.append(node)
    
    # If we still don't have suitable nodes, include nodes with certain edge relations
    if not suitable_nodes:
        for source, target, data in nx_graph.edges(data=True):
            label = data.get('label', '')
            if ':ARG' in label or ':mod' in label:
                suitable_nodes.append(source)
    
    return list(set(suitable_nodes))  # Remove duplicates

def _add_extraneous_node(
    nx_graph: nx.DiGraph, 
    target_node: str, 
    concept_type: str, 
    concept_value: str
) -> Tuple[str, str]:
    """
    Add an extraneous node to the graph connected to the target node.
    
    Args:
        nx_graph: The NetworkX graph
        target_node: The node to connect to
        concept_type: The type of concept to add
        concept_value: The value for the concept
        
    Returns:
        A tuple containing the new node ID and the relation used
    """
    # Generate a new node ID
    existing_nodes = list(nx_graph.nodes())
    max_id = 0
    for node in existing_nodes:
        if node.startswith('z') and node[1:].isdigit():
            max_id = max(max_id, int(node[1:]))
    
    new_node_id = f"z{max_id + 1}"
    
    # Determine appropriate relation based on concept type
    relation_options = {
        'person': [':mod', ':ARG0-of', ':ARG1-of', ':poss', ':beneficiary', ':accompanier'],
        'organization': [':mod', ':org', ':ARG0-of', ':employer', ':affiliation'],
        'location': [':location', ':mod', ':destination', ':source', ':path'],
        'date-entity': [':time', ':mod', ':duration', ':frequency', ':dayperiod'],
        'quantity': [':quant', ':mod', ':value', ':scale', ':unit'],
        'event': [':mod', ':topic', ':manner', ':purpose', ':cause']
    }
    
    # Select a relation appropriate for this concept type
    relations = relation_options.get(concept_type, [':mod'])
    relation = random.choice(relations)
    
    # Add the concept node
    nx_graph.add_node(new_node_id, instance=f"{concept_type}/")
    
    # Connect it to the target node
    nx_graph.add_edge(target_node, new_node_id, label=relation)
    
    # For person, organization, and location, add a name structure
    if concept_type in ['person', 'organization', 'location']:
        name_node_id = f"z{max_id + 2}"
        nx_graph.add_node(name_node_id, instance='name/')
        nx_graph.add_edge(new_node_id, name_node_id, label=':name')
        
        # Split into name parts for a more realistic AMR structure
        name_parts = concept_value.split()
        for i, part in enumerate(name_parts[:3]):  # Limit to 3 name parts
            nx_graph.add_edge(name_node_id, f'"{part}"', label=f':op{i+1}')
    
    # For date-entity, add appropriate attributes
    elif concept_type == 'date-entity':
        # Try to extract year, month, day if present
        if 'tahun' in concept_value:
            year_match = concept_value.split('tahun ')[1].strip() if 'tahun ' in concept_value else ''
            if year_match.isdigit():
                nx_graph.add_edge(new_node_id, f'"{year_match}"', label=':year')
        elif any(month in concept_value.lower() for month in ['januari', 'februari', 'maret', 'april', 'mei', 'juni', 'juli', 'agustus', 'september', 'oktober', 'november', 'desember']):
            # Create a proper date-entity node for the month value
            date_node_id = f"z{max_id + 2}"
            nx_graph.add_node(date_node_id, instance='date-entity/')
            nx_graph.add_edge(new_node_id, date_node_id, label=':month')
            nx_graph.add_edge(date_node_id, f'"{concept_value}"', label=':value')
        else:
            # Create a proper date-entity node for the time value
            date_node_id = f"z{max_id + 2}"
            nx_graph.add_node(date_node_id, instance='date-entity/')
            nx_graph.add_edge(new_node_id, date_node_id, label=':time')
            nx_graph.add_edge(date_node_id, f'"{concept_value}"', label=':value')
    
    # For quantity, add value directly
    elif concept_type == 'quantity':
        # Create a proper quantity node for the value
        quantity_node_id = f"z{max_id + 2}"
        nx_graph.add_node(quantity_node_id, instance='quantity/')
        nx_graph.add_edge(new_node_id, quantity_node_id, label=':value')
        nx_graph.add_edge(quantity_node_id, f'"{concept_value}"', label=':quant')
    
    # For other types, add a generic mod value
    else:
        # Create a proper entity node for the value
        entity_node_id = f"z{max_id + 2}"
        nx_graph.add_node(entity_node_id, instance=f"{concept_type}/")
        nx_graph.add_edge(new_node_id, entity_node_id, label=':mod')
        nx_graph.add_edge(entity_node_id, f'"{concept_value}"', label=':value')
    
    return new_node_id, relation
