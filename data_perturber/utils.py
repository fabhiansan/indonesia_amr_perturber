import re
from networkx import DiGraph
import penman
from penman import Graph
import networkx as nx
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from deep_translator import GoogleTranslator
import random

pattern = r"\([a-z][0-9][\s\S]*"

def return_amr(amr_string):
    """
    Returns the AMR string from the given string
    """
    if not amr_string:
        raise ValueError("Empty AMR string provided")
        
    match = re.search(pattern, amr_string)
    if not match:
        raise ValueError(f"Could not extract AMR from string: {amr_string[:100]}...")
    return match.group(0)
    
    
def penman_to_networkx(penman_graph: Graph) -> DiGraph:
    """
    Converts a Penman graph to a NetworkX graph.
    
    Args:
        penman_graph: A penman.Graph object
        
    Returns:
        A NetworkX DiGraph object
    """
    G = nx.DiGraph()  # Use a directed graph (or nx.Graph() for undirected)
    for source, relation, target in penman_graph.triples:
        # Optionally add nodes explicitly if you want to attach more attributes later
        if source not in G:
            G.add_node(source)
        if target not in G:
            G.add_node(target)
        # Add an edge with the relation as an attribute
        G.add_edge(source, target, label=relation)
    return G

def networkx_to_penman(G, top=None):
    """
    Converts a NetworkX graph to a Penman graph.
    
    Args:
        G: A NetworkX DiGraph object
        top: Optional top node for the Penman graph
        
    Returns:
        A Penman Graph object
    """
    # Reconstruct triples from the NetworkX graph.
    triples = []
    for source, target, data in G.edges(data=True):
        # The relation label is expected to be stored in the 'label' attribute.
        relation = data.get('label', '')
        triples.append((source, relation, target))
    
    # If no top node is provided, try to find one (for directed graphs, a node with no incoming edges)
    if top is None:
        if G.is_directed():
            for node in G.nodes():
                if G.in_degree(node) == 0:
                    top = node
                    break
            if top is None:
                # Fallback if every node has an incoming edge
                top = next(iter(G.nodes()))
        else:
            top = next(iter(G.nodes()))
    
    # Create and return the Penman graph
    return penman.Graph(triples, top=top)


def get_indonesian_antonyms(word, max_antonyms=5):
    """
    Get antonyms for an Indonesian word by:
    1. Finding antonyms using WordNet
    2. Translating antonyms back to Indonesian
    
    Args:
        word (str): Indonesian word to find antonyms for
        max_antonyms (int): Maximum number of antonyms to return
        
    Returns:
        list: List of Indonesian antonyms
    """    
    # Step 1: Find antonyms using WordNet
    english_antonyms = []
    for synset in wn.synsets(word, lang='ind'):
        for lemma in synset.lemmas():
            if lemma.antonyms():
                for antonym in lemma.antonyms():
                    english_antonyms.append(antonym.name().replace('_', ' '))
    
    if not english_antonyms:
        print(f"No antonyms found for '{word}'")
        return []
    
    # Limit number of antonyms to process
    english_antonyms = english_antonyms[:max_antonyms]
    # print(f"Found English antonyms: {english_antonyms}")
    
    # Step 2: Translate antonyms back to Indonesian
    indonesian_antonyms = []
    translator = GoogleTranslator(source='en', target='id')
    
    for antonym in english_antonyms:
        try:
            indonesian_antonym = translator.translate(antonym)
            indonesian_antonyms.append(indonesian_antonym)
        except Exception as e:
            print(f"Error translating '{antonym}': {str(e)}")
    print(f"for word {word} \n Found Indonesian antonyms: {indonesian_antonyms}")
    return indonesian_antonyms

def get_one_indonesian_antonyms(word):
    return random.choice(get_indonesian_antonyms(word))
