from penman import Graph
from .utils import penman_to_networkx, networkx_to_penman
import random
import networkx as nx

def is_person_or_agent(graph, node):
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

def EntityError(amr_graph: Graph):
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
    """
    nx_gr = penman_to_networkx(amr_graph)
    
    # Find valid ARG0-ARG1 pairs to swap (where both are people/agents)
    valid_predicates = []
    
    # Group edges by source (predicate)
    predicate_args = {}
    for u, v, data in nx_gr.edges(data=True):
        if data.get('label') in [':ARG0', ':ARG1']:
            if u not in predicate_args:
                predicate_args[u] = {}
            predicate_args[u][data.get('label')] = v
    
    # Check predicates with both ARG0 and ARG1
    for pred, args in predicate_args.items():
        if ':ARG0' in args and ':ARG1' in args:
            arg0 = args[':ARG0']
            arg1 = args[':ARG1']
            
            # Only swap if both are people/agents
            if is_person_or_agent(nx_gr, arg0) and is_person_or_agent(nx_gr, arg1):
                valid_predicates.append(pred)
    
    # If we have valid predicates to modify, choose one randomly
    if valid_predicates:
        chosen_pred = random.choice(valid_predicates)
        arg0 = predicate_args[chosen_pred][':ARG0']
        arg1 = predicate_args[chosen_pred][':ARG1']
        
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
        # Fallback: do the general swap if no valid people/agent pairs found
        for u, v, data in nx_gr.edges(data=True):
            if data.get('label') == ':ARG0':
                data['label'] = ':ARG1'
            elif data.get('label') == ':ARG1':
                data['label'] = ':ARG0'
            
    return networkx_to_penman(nx_gr)

def change_quant_source(G, old_source, new_source, label=':quant'):
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

def ensure_connected(G):
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