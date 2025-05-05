from typing import Tuple, Dict, Any
import random
from penman import Graph
import networkx as nx
from .utils import penman_to_networkx, networkx_to_penman
from faker import Faker
from .circumstance_perturber import get_entity_category # Reusing categorization logic
import logging
fake = Faker('id_ID') # Initialize Faker with Indonesian locale

def is_person_or_agent(graph: nx.DiGraph, node: str) -> bool:
    """
    Check if a node represents a person or an agent-like entity.
    
    Args:
        graph: NetworkX graph
        node: Node to check
        
    Returns:
        bool: True if node likely represents a person or agent, False otherwise
    """
    # Common person/agent concept patterns
    person_patterns = ['person', 'orang', 'personil', 'manusia', 'pengguna', 
                       'kelompok', 'grup', 'bangsa', 'individu', 'pasien',
                       'organisasi', 'pemerintah', 'gerakan', 'perusahaan']
    
    # Check instance type if available
    for u, v, data in graph.edges(data=True):
        if v == node and data.get('label') == ':instance':
            instance_type = u.lower()
            # Check if the instance type matches any person pattern
            if any(pattern in instance_type for pattern in person_patterns):
                return True
    
    # Check for name attribute (entities with names are often people/organizations)
    has_name = False
    for u, v, data in graph.edges(data=True):
        if u == node and data.get('label') == ':name':
            has_name = True
            break
    
    # Check for other agent-like properties
    agent_properties = [':ARG0-of', ':poss', ':beneficiary', ':accompanier', ':topic']
    for u, v, data in graph.edges(data=True):
        if u == node and any(prop == data.get('label') for prop in agent_properties):
            return True
    
    return has_name  # If it has a name but no other indicators, still consider it a potential agent

def insertEntityError(amr_graph: Graph) -> Tuple[Graph, Dict[str, Any]]:
    """
    Entity Error. Entity errors manifest when the
    entities associated with a predicate in a summary
    are incorrectly attributed or erroneous. These errors are crafted through two principal sources: 
    
    1. By swapping the roles of the agent and the patient,
       which results in the misattribution of actions or characteristics
    2. By substituting specific entities, such as names and numbers
    
    In AMR graphs, the clear distinction between agent (ARG0) and patient (ARG1) 
    allows for straightforward swaps. We implement agent-patient swaps by exchanging 
    the roles of the agent and the patient. Here, the agent refers to an action doer, 
    and the patient refers to an action recipient.
    
    This implementation checks if both ARG0 and ARG1 are people or agent-like entities
    before performing the swap to ensure semantic plausibility.
    
    Args:
        amr_graph: AMR graph in Penman format

    Returns:
        A tuple of (perturbed_graph, changelog) where:
        - perturbed_graph is the modified AMR graph
        - changelog is a dictionary describing the changes made
    """
    # Preserve original graph and get its top node, then convert to NetworkX
    original_top = amr_graph.top
    nx_gr = penman_to_networkx(amr_graph)
    logger = logging.getLogger(__name__) # Define logger inside function scope
    
    # Find predicates that have both :ARG0 and :ARG1 edges
    potential_preds = []
    for node in nx_gr.nodes():
        arg0 = None
        arg1 = None
        # Check outgoing edges for :ARG0 and :ARG1
        for _, neighbor, data in nx_gr.out_edges(node, data=True):
            if data.get('label') == ':ARG0':
                arg0 = neighbor
            elif data.get('label') == ':ARG1':
                arg1 = neighbor
        
        # If the predicate has both :ARG0 and :ARG1, and both are people/agents, add to potential_preds
        if arg0 and arg1 and is_person_or_agent(nx_gr, arg0) and is_person_or_agent(nx_gr, arg1):
            potential_preds.append((node, arg0, arg1))
    
    # If there are potential predicates to modify, choose one randomly
    if potential_preds:
        chosen_pred, arg0, arg1 = random.choice(potential_preds)
        
        # Track changes for changelog
        changelog = {
            'type': 'entity_error',
            'description': 'Swapped agent and patient roles',
            'predicate': chosen_pred,
            'swapped_entities': {
                'ARG0': arg0,
                'ARG1': arg1
            }
        }
        
        # Swap ARG0 and ARG1
        for u, v, data in list(nx_gr.edges(data=True)):
            if u == chosen_pred:
                if data.get('label') == ':ARG0':
                    nx_gr.remove_edge(u, v)
                    nx_gr.add_edge(u, arg1, label=':ARG0')
                elif data.get('label') == ':ARG1':
                    nx_gr.remove_edge(u, v)
                    nx_gr.add_edge(u, arg0, label=':ARG1')
    else:
        logger.debug("No potential predicates for swap found. Attempting entity substitution.")
        logger.debug("No potential predicates for swap found. Attempting entity substitution.")

        # Collect all name entities (edges to literal values)
        name_entities = []
        for u, v, data in nx_gr.edges(data=True):
            if data.get('label') == ':name':
                for _, literal_v, literal_data in list(nx_gr.out_edges(v, data=True)): # Use list() to allow modification during iteration
                    if literal_data.get('label', '').startswith(':op') and isinstance(literal_v, str) and literal_v.startswith('"'):
                         logger.debug(f"Found potential name entity: {v} -> {literal_v}")
                         # Store the edge info: (source_node, literal_value_node, edge_data)
                         name_entities.append((v, literal_v, literal_data))

        logger.debug(f"Found {len(name_entities)} name entities for potential swap.")

        # Strategy 2a: Swap existing name entities if at least two are found
        if len(name_entities) >= 2:
            logger.debug("Attempting to swap two name entities.")
            # Randomly select two distinct name entities
            entity1_info, entity2_info = random.sample(name_entities, 2)
            source1, value_node1, edge_data1 = entity1_info
            source2, value_node2, edge_data2 = entity2_info

            original_value1 = value_node1.strip('"')
            original_value2 = value_node2.strip('"')

            logger.debug(f"Swapping '{original_value1}' (from {source1}) with '{original_value2}' (from {source2})")

            # Swap the literal values by modifying the edges
            # Remove old edges
            nx_gr.remove_edge(source1, value_node1)
            nx_gr.remove_edge(source2, value_node2)

            # Add new edges with swapped values
            nx_gr.add_edge(source1, value_node2, label=edge_data1.get('label'))
            nx_gr.add_edge(source2, value_node1, label=edge_data2.get('label'))

            changelog = {
                'type': 'entity_error',
                'description': 'Swapped two name entity values',
                'swapped_entities': [
                    {'node1': source1, 'original_value1': original_value1, 'new_value1': original_value2},
                    {'node2': source2, 'original_value2': original_value2, 'new_value2': original_value1}
                ]
            }
            logger.debug(f"Name entity swap successful. Changelog: {changelog}")

        # Strategy 2b: Substitute a single entity if name swap is not possible
        else:
            logger.debug("Fewer than 2 name entities found. Attempting single entity substitution.")
            # Collect quantity entities
            quant_entities = []
            for u, v, data in nx_gr.edges(data=True):
                 if data.get('label') == ':quant':
                      if isinstance(v, str) and v.startswith('"'):
                         logger.debug(f"Found potential quant entity: {u} -> {v}")
                         quant_entities.append((u, v, data, 'quant')) # Store (source_node, literal_value_node, edge_data, type)

            # Combine remaining name entities (if any) and quantity entities
            # Note: name_entities here are the ones that couldn't be part of a swap
            potential_single_substitution_entities = []
            for source, value_node, edge_data in name_entities: # Add name entities that weren't swapped
                 potential_single_substitution_entities.append((source, value_node, edge_data, 'name'))
            potential_single_substitution_entities.extend(quant_entities) # Add quantity entities

            logger.debug(f"Potential entities for single substitution: {potential_single_substitution_entities}")

            if potential_single_substitution_entities:
                # Randomly select one entity for single substitution
                chosen_source, original_value_node, edge_data, entity_type = random.choice(potential_single_substitution_entities)
                original_value = original_value_node.strip('"') # Remove quotes
                logger.debug(f"Selected entity for single substitution: {original_value} ({entity_type})")

                # Generate a replacement value
                new_value = get_general_entity_replacement(original_value, entity_type)
                new_value_node = f'"{new_value}"' # Add quotes back for literal
                logger.debug(f"Generated new value: {new_value}")

                # Modify the graph
                if entity_type == 'name':
                    # Replace the specific :opN edge target
                    nx_gr.remove_edge(chosen_source, original_value_node)
                    nx_gr.add_edge(chosen_source, new_value_node, label=edge_data.get('label'))
                elif entity_type == 'quant':
                     # Replace the :quant edge target
                     nx_gr.remove_edge(chosen_source, original_value_node)
                     nx_gr.add_edge(chosen_source, new_value_node, label=':quant')

                changelog = {
                    'type': 'entity_error',
                    'description': f'Substituted single {entity_type} entity value',
                    'original_value': original_value,
                    'new_value': new_value,
                    'modified_node': chosen_source # Node the attribute was attached to
                }
                logger.debug(f"Single entity substitution successful. Changelog: {changelog}")

            else:
                logger.debug("No potential entities for substitution found. Returning no_change.")
                # If neither ARG0/ARG1 swap nor any entity substitution is possible
                nx_gr = ensure_connected(nx_gr) # Keep ensure_connected here
                changelog = {
                    'type': 'entity_error',
                    'description': 'No suitable entities found for swapping or substitution',
                    'action': 'no_change'
                }

    # Convert back to Penman, preserving original top node
    return networkx_to_penman(nx_gr, top=original_top), changelog

def change_quant_source(G: nx.DiGraph, old_source: str, new_source: str, label: str = ':quant') -> nx.DiGraph:
    """
    Mengganti sumber (source) dari edge yang memiliki label tertentu.
    Pada kasus ini, mengganti edge dengan label ':quant' yang awalnya dari old_source 
    menjadi dari new_source.
    """
    # Kumpulkan edge yang akan diubah
    edges_to_modify = []
    for u, v, data in list(G.edges(data=True)):
        if u == old_source and data.get('label') == label:
            edges_to_modify.append((u, v, data))
    
    # Lakukan perubahan: hapus edge lama dan tambahkan edge baru dengan sumber baru
    for u, v, data in edges_to_modify:
        G.remove_edge(u, v)
        # Pastikan new_source ada di graf
        if new_source not in G:
            G.add_node(new_source)
        G.add_edge(new_source, v, label=data.get('label'))
    
    return G

def ensure_connected(G: nx.DiGraph) -> nx.DiGraph:
    """
    Memastikan graf (directed) tetap terhubung secara lemah (weakly connected)
    sehingga saat konversi ke Penman tidak terjadi LayoutError karena graf terputus.
    Jika ditemukan komponen yang terpisah, fungsi ini akan menghubungkannya dengan
    menambahkan edge dari top node ke salah satu node di komponen tersebut.
    """
    if not nx.is_weakly_connected(G):
        # Cari node top (node dengan in_degree = 0)
        top = None
        for node in G.nodes():
            if G.in_degree(node) == 0:
                top = node
                break
        if top is None:
            top = next(iter(G.nodes()))
        
        # Dapatkan semua komponen terhubung secara lemah
        components = list(nx.weakly_connected_components(G))
        print('weak graph', components)
        # Komponen utama adalah yang memuat top node
        main_component = None
        for comp in components:
            if top in comp:
                main_component = comp
                break
        # Untuk setiap komponen yang terpisah, tambahkan edge dari top ke salah satu nodenya
        for comp in components:
            if comp != main_component:
                node_in_comp = next(iter(comp))
                G.add_edge(top, node_in_comp, label=':link')
    return G

def get_general_entity_replacement(entity_value: str, entity_type: str) -> str:
    """
    Generates a plausible replacement value for a given entity value and type.
    """
    if entity_type == "name":
        # Generate a random Indonesian name
        return fake.name()
    elif entity_type == "quant":
        # Generate a random number (e.g., between 1 and 100)
        return str(random.randint(1, 100))
    # Add more types if needed in the future
    else:
        # Fallback to a generic word
        return fake.word()
