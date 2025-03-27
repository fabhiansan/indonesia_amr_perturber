from typing import Tuple, Dict, Any
import random
from penman import Graph
import networkx as nx
from .utils import penman_to_networkx, networkx_to_penman

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

def EntityError(amr_graph: Graph) -> Tuple[Graph, Dict[str, Any]]:
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
    # Convert Penman graph to NetworkX for easier manipulation
    nx_gr = penman_to_networkx(amr_graph)
    
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
        nx_gr = ensure_connected(nx_gr)
        changelog = {
            'type': 'entity_error',
            'description': 'No suitable entities found for swapping',
            'action': 'no_change'
        }
        
    return networkx_to_penman(nx_gr), changelog

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
