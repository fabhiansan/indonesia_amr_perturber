import penman
from data_perturber.utils import return_amr
from data_perturber.predicates_perturber import insertWrongPredicates
from data_perturber.circumstance_perturber import insertCircumstanceError
from data_perturber.entity_perturber import EntityError
from data_perturber.discourse_perturber import insertDiscourseError
from data_perturber.out_of_article_perturber import insertOutOfArticleError

def predicate_error_insertion(amr_string: str) -> tuple[penman.Graph, list]:
    """
    Insert n_wrong wrong predicted nodes into an AMR graph.
    
    Args:
        amr_string (str): AMR graph string
        n_wrong (int): Number of wrong predicted nodes to insert
        
    Returns:
        tuple: (perturbed_graph, changelog)
    """
    # data cleaning, removing meta data using regex
    amr_string = return_amr(amr_string)
    
    # graph creationg
    gr = penman.decode(amr_string)
    
    # insert wrong preds, return graph list and changelog
    gr2, cl = insertWrongPredicates(gr)
    
    return gr2, cl

def circumstance_error_insertion(amr_string: str, error_type: str = "both") -> tuple[penman.Graph, list]:
    """
    Insert circumstance errors into an AMR graph.
    
    Circumstance errors occur when there is incorrect information regarding context
    (location, time, modality) of predicate interactions. This function creates two types
    of circumstance errors:
    1. Modality intensification - altering degree of certainty in statements
    2. Substitution of circumstance entities - replacing locations, times, etc.
    
    Args:
        amr_string (str): AMR graph string
        error_type (str): Type of error to insert - "modality", "entity", or "both"
        
    Returns:
        tuple: (perturbed_graph, changelog)
    """
    # Clean AMR string, removing meta data
    amr_string = return_amr(amr_string)
    
    # Decode AMR string to graph
    gr = penman.decode(amr_string)
    
    # Insert circumstance errors, return graph list and changelog
    gr2, cl = insertCircumstanceError(gr, error_type)
    
    # Generate text from original and perturbed graphs
    return gr2, cl

def entity_error_insertion(amr_string: str) -> penman.Graph:
    """
    Insert entity errors into an AMR graph.
    
    Entity errors occur when entities associated with a predicate are incorrectly 
    attributed or erroneous. This function implements agent-patient swaps by exchanging
    the roles of ARG0 and ARG1, focusing on cases where both arguments are people
    or agent-like entities.
    
    Args:
        amr_string (str): AMR graph string
        
    Returns:
        AMRGraph: The perturbed AMR graph
    """
    # Clean AMR string, removing meta data
    amr_string = return_amr(amr_string)
    
    # Decode AMR string to graph
    gr = penman.decode(amr_string)
    
    # Apply entity error perturbation
    perturbed_gr = EntityError(gr)
    
    # Generate text from original and perturbed graphs
    return perturbed_gr

def discourse_error_insertion(amr_string: str) -> tuple[penman.Graph, list]:
    """
    Insert discourse link errors into an AMR graph.
    
    Discourse errors occur when temporal/causal relations between events are incorrect.
    
    Args:
        amr_string (str): AMR graph string
        
    Returns:
        tuple: (perturbed_graph, changelog)
    """
    amr_string = return_amr(amr_string)
    gr = penman.decode(amr_string)
    gr2, cl = insertDiscourseError(gr)
    return gr2, cl

def out_of_article_error_insertion(amr_string: str) -> tuple[penman.Graph, list]:
    """
    Insert out-of-article errors into an AMR graph.
    
    Out-of-article errors occur when extraneous information not from the original
    article is added to the graph.
    
    Args:
        amr_string (str): AMR graph string
        
    Returns:
        tuple: (perturbed_graph, changelog)
    """
    amr_string = return_amr(amr_string)
    gr = penman.decode(amr_string)
    gr2, cl = insertOutOfArticleError(gr)
    return gr2, cl
