import penman
from penman import Graph
import random
import networkx as nx
from typing import Tuple, List, Dict, Any, Optional, Union
from data_perturber.utils import penman_to_networkx, networkx_to_penman, get_indonesian_antonyms

def insertCircumstanceError(amr_graph: Graph, error_type: str = "both") -> Tuple[Graph, List[Dict[str, str]]]:
    """
    Circumstance Error. Circumstance errors in summaries emerge when there is incorrect 
    or misleading information regarding the context of predicate interactions, specifically 
    in terms of location, time, and modality.
    
    These errors are mainly created in two ways:
    1. By intensifying the modality, which alters the degree of certainty or possibility
       expressed in the statement
    2. By substituting specific entities like locations and times
    
    Args:
        amr_graph: AMR graph in Penman format
        error_type: Type of error to insert - "modality", "entity", or "both"
        
    Returns:
        A tuple of (perturbed_graph, changelog) where:
        - perturbed_graph is the modified AMR graph
        - changelog is a list of dictionaries describing the changes made
    """
    perturbed_graphs: List[Graph] = []
    changelog: List[Dict[str, str]] = []
    
    # Convert to networkx for manipulation
    nx_graph: nx.DiGraph = penman_to_networkx(amr_graph)
    
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

def modify_modality(nx_graph: nx.DiGraph) -> Tuple[List[Graph], List[Dict[str, str]]]:
    """
    Modify modality in AMR graph by replacing modality concepts with stronger/weaker ones.
    
    Args:
        nx_graph: NetworkX graph representation of AMR
        
    Returns:
        A tuple of (perturbed_graphs, changelog) where:
        - perturbed_graphs is a list of modified AMR graphs
        - changelog is a list of dictionaries describing the changes made
    """
    perturbed_graphs: List[Graph] = []
    changelog: List[Dict[str, str]] = []
    
    # Common Indonesian modality concepts and their intensified versions
    # Format: {original: intensified}
    modality_mapping: Dict[str, str] = {
        "mungkin": "harus",         # possible → must
        "dapat": "wajib",           # can → obligatory 
        "bisa": "pasti",            # can → definitely
        "boleh": "wajib",           # may → obligatory
        "izin": "perintah",         # permission → command
        "ragu": "yakin",            # doubt → certain
        "pikir": "tahu",            # think → know
        "usul": "menuntut",         # suggest → demand
        "minta": "perintah",        # ask → command
        "permohonan": "persyaratan", # request → requirement
        "coba": "berhasil",         # try → succeed
        "kadang": "selalu",         # sometimes → always
        "semoga": "pasti",          # hopefully → definitely
        "kira": "pasti",            # guess → certain
        "kemungkinan": "kepastian", # possibility → certainty
        "seharusnya": "wajib",      # should → must
        "ingin": "butuh",           # want → need
        "harap": "tuntut",          # hope → demand
        "diperkirakan": "dipastikan", # estimated → confirmed
        "berpotensi": "pasti",      # potential → definite
        "mencoba": "berhasil",      # try → succeed
        "agak": "sangat",           # somewhat → very
        "sementara": "permanen",    # temporary → permanent
        "diizinkan": "diwajibkan"   # allowed → required
    }
    
    # Find all nodes that may represent modality
    for node in list(nx_graph.nodes()):
        # Check if node is a string value (concept) that exists in our mapping
        if isinstance(node, str) and any(modal in node.lower() for modal in modality_mapping.keys()):
            # Get the base concept (without instance number)
            if "-" in node:
                base_concept: str = node.split("-")[0].lower()
            else:
                base_concept: str = node.lower()
            
            # Find matching concepts to replace
            for original, intensified in modality_mapping.items():
                if original in base_concept:
                    # Create a new graph with the replaced modality
                    if "-" in node:
                        # Preserve the instance number
                        instance_num: str = node.split("-")[1]
                        new_node: str = f"{intensified}-{instance_num}"
                    else:
                        new_node: str = intensified
                    
                    # Create a copy to avoid modifying the original
                    graph_copy: nx.DiGraph = nx_graph.copy()
                    
                    # Relabel the node
                    graph_copy = nx.relabel_nodes(graph_copy, {node: new_node})
                    
                    # Convert back to penman
                    penman_graph: Graph = networkx_to_penman(graph_copy)
                    
                    perturbed_graphs.append(penman_graph)
                    changelog.append({f"Changed modality": f"'{node}' → '{new_node}'"})
                    break  # Only apply one replacement per node
    
    return perturbed_graphs, changelog

def substitute_circumstance_entities(nx_graph: nx.DiGraph) -> Tuple[List[Graph], List[Dict[str, str]]]:
    """
    Substitute circumstance entities in AMR graph by identifying and replacing
    nodes related to time, location, and other circumstantial elements.
    
    Args:
        nx_graph: NetworkX graph representation of AMR
        
    Returns:
        A tuple of (perturbed_graphs, changelog) where:
        - perturbed_graphs is a list of modified AMR graphs
        - changelog is a list of dictionaries describing the changes made
    """
    perturbed_graphs: List[Graph] = []
    changelog: List[Dict[str, str]] = []
    
    # Circumstance relations in AMR grouped by category
    time_relations: List[str] = [
        ":time", ":time-of", ":duration", ":decade", ":year", ":month", ":day", 
        ":weekday", ":dayperiod", ":season", ":timezone"
    ]
    
    location_relations: List[str] = [
        ":location", ":source", ":destination", ":path", ":direction"
    ]
    
    causal_relations: List[str] = [
        ":purpose", ":cause", ":condition", ":concession"
    ]
    
    manner_relations: List[str] = [
        ":manner", ":compared-to", ":extent"
    ]
    
    # All circumstance relations combined
    circumstance_relations: List[str] = time_relations + location_relations + causal_relations + manner_relations
    
    # Map relations to their category for easier substitution
    relation_categories: Dict[str, List[str]] = {
        "time": time_relations,
        "location": location_relations,
        "causal": causal_relations,
        "manner": manner_relations
    }
    
    # Find all edges with circumstance relations
    circumstance_edges: List[Tuple[str, str, Dict[str, str]]] = []
    for u, v, data in nx_graph.edges(data=True):
        if data.get('label') in circumstance_relations:
            circumstance_edges.append((u, v, data))
    
    if not circumstance_edges:
        return perturbed_graphs, changelog  # No circumstance entities to modify
    
    # Randomly select one circumstance edge to modify
    if circumstance_edges:
        edge_to_modify: Tuple[str, str, Dict[str, str]] = random.choice(circumstance_edges)
        source, target, data = edge_to_modify
        relation: str = data.get('label')
        
        # Create a modified graph copy
        graph_copy: nx.DiGraph = nx_graph.copy()
        
        # STRATEGY 1: Replace the entity with a different but plausible entity
        if isinstance(target, str) and not target.startswith(":"):
            # For named entities, we could change them to different names
            if "-" in target:
                base_entity: str = target.split("-")[0]
                # Get an "antonym" or different entity
                replacements: List[str] = get_entity_replacements(base_entity)
                if replacements:
                    new_entity: str = random.choice(replacements)
                    # Preserve instance number
                    instance_num: str = target.split("-")[1]
                    new_target: str = f"{new_entity}-{instance_num}"
                    
                    # Relabel the node
                    graph_copy = nx.relabel_nodes(graph_copy, {target: new_target})
                    
                    # Convert back to penman
                    penman_graph: Graph = networkx_to_penman(graph_copy)
                    
                    perturbed_graphs.append(penman_graph)
                    changelog.append({f"Changed {relation[1:]} entity": f"'{target}' → '{new_target}'"})
        
        # STRATEGY 2: Change relation type (more diverse changes)
        if not perturbed_graphs:  # If first strategy didn't work
            # Determine which category the relation belongs to
            category: Optional[str] = None
            for cat, rels in relation_categories.items():
                if relation in rels:
                    category = cat
                    break
            
            if category:
                # Choose a new relation from the same category, but different from current
                possible_relations: List[str] = [r for r in relation_categories[category] if r != relation]
                
                if possible_relations:
                    new_relation: str = random.choice(possible_relations)
                    
                    # Create a copy and modify the edge
                    graph_copy: nx.DiGraph = nx_graph.copy()
                    graph_copy.remove_edge(source, target)
                    graph_copy.add_edge(source, target, label=new_relation)
                    
                    # Convert back to penman
                    penman_graph: Graph = networkx_to_penman(graph_copy)
                    
                    perturbed_graphs.append(penman_graph)
                    changelog.append({'type': 'discourse_error', 'old_relation': relation, 'new_relation': new_relation, 'nodes': (source, target)})
            else:
                # If current relation doesn't fit a category, choose a random category and relation
                random_category: str = random.choice(list(relation_categories.keys()))
                new_relation: str = random.choice(relation_categories[random_category])
                
                # Create a copy and modify the edge
                graph_copy: nx.DiGraph = nx_graph.copy()
                graph_copy.remove_edge(source, target)
                graph_copy.add_edge(source, target, label=new_relation)
                
                # Convert back to penman
                penman_graph: Graph = networkx_to_penman(graph_copy)
                
                perturbed_graphs.append(penman_graph)
                changelog.append({'type': 'discourse_error', 'old_relation': relation, 'new_relation': new_relation, 'nodes': (source, target)})
        
        # STRATEGY 3: Add temporal or locative specificity (for basic relations)
        if not perturbed_graphs and relation in [":time", ":location"]:
            # More specific temporal/locative relations
            specific_time_relations: List[str] = [":decade", ":year", ":month", ":day", ":weekday"]
            specific_location_relations: List[str] = [":source", ":destination", ":path"]
            
            if relation == ":time":
                new_relation: str = random.choice(specific_time_relations)
            else:  # relation == ":location"
                new_relation: str = random.choice(specific_location_relations)
            
            # Create a copy and modify the edge
            graph_copy: nx.DiGraph = nx_graph.copy()
            graph_copy.remove_edge(source, target)
            graph_copy.add_edge(source, target, label=new_relation)
            
            # Convert back to penman
            penman_graph: Graph = networkx_to_penman(graph_copy)
            
            perturbed_graphs.append(penman_graph)
            changelog.append({'type': 'discourse_error', 'old_relation': relation, 'new_relation': new_relation, 'nodes': (source, target)})
    
    return perturbed_graphs, changelog

def get_entity_replacements(entity: str, n: int = 3) -> List[str]:
    """
    Get replacement entities based on concept type using Faker and NLTK WordNet categories.
    Generates realistic Indonesian alternatives for various entity types.
    
    Args:
        entity: Original entity to replace
        n: Number of replacements to generate
        
    Returns:
        List of possible replacement entities
    """
    from faker import Faker
    import random
    from nltk.corpus import wordnet as wn
    
    # Create Faker instance with Indonesian locale
    fake = Faker('id_ID')
    
    # Initialize entity categorization
    category = get_entity_category(entity)
    
    replacements: List[str] = []
    
    # Generate appropriate replacements based on entity category
    if category == "TIME":
        # Time-related replacements
        for _ in range(n):
            time_options = [
                fake.day_of_week(),
                fake.month_name(),
                fake.date_of_birth().strftime("%d %B %Y"),
                fake.time_of_day(),
                fake.future_date(end_date="+30d").strftime("%d %B %Y")
            ]
            replacements.append(random.choice(time_options))
    
    elif category == "LOCATION":
        # Location-related replacements
        for _ in range(n):
            location_options = [
                fake.city(),
                fake.province(),
                fake.street_name(),
                fake.building_name(),
                fake.country(),
                fake.administrative_unit()
            ]
            replacements.append(random.choice(location_options))
    
    elif category == "PERSON":
        # Person-related replacements
        for _ in range(n):
            person_options = [
                fake.name(),
                fake.first_name(),
                fake.last_name(),
                fake.name_male(),
                fake.name_female(),
                f"{fake.prefix()} {fake.name()}"
            ]
            replacements.append(random.choice(person_options))
    
    elif category == "ORGANIZATION":
        # Organization-related replacements
        for _ in range(n):
            org_options = [
                f"PT {fake.company()}",
                f"{fake.company()} (Persero) Tbk",
                f"Yayasan {fake.last_name()}",
                f"Universitas {fake.last_name()}",
                f"CV {fake.last_name()}",
                f"Kementerian {fake.catch_phrase_noun()}"
            ]
            replacements.append(random.choice(org_options))
    
    else:
        # Generic entities or fallback
        for _ in range(n):
            generic_options = [
                fake.word(),
                fake.catch_phrase_noun(),
                fake.bs_noun(),
                fake.currency_name(),
                fake.file_name()
            ]
            replacements.append(random.choice(generic_options))
    
    # Remove duplicates and filter out the original entity
    replacements = list(set([r for r in replacements if r.lower() != entity.lower()]))
    
    # Ensure we return at most n items
    return replacements[:n] if len(replacements) > n else replacements

def get_entity_category(entity: str) -> str:
    """
    Determine the semantic category of an entity using NLP techniques.
    
    Args:
        entity: The entity to categorize
        
    Returns:
        Category string: "TIME", "LOCATION", "PERSON", "ORGANIZATION", or "OTHER"
    """
    entity_lower = entity.lower()
    
    # Use lexical matching with categorized keyword lists
    
    # Time-related entities
    time_keywords = {
        # Indonesian days
        "senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu",
        # Indonesian months
        "januari", "februari", "maret", "april", "mei", "juni", "juli", 
        "agustus", "september", "oktober", "november", "desember",
        # Time units and periods
        "hari", "minggu", "bulan", "tahun", "jam", "menit", "detik", 
        "pagi", "siang", "sore", "malam", "waktu", "durasi", "musim",
        # Time references
        "kemarin", "besok", "lusa", "tadi", "nanti", "sekarang"
    }
    
    # Location-related entities
    location_keywords = {
        # Place types
        "tempat", "lokasi", "daerah", "wilayah", "kawasan", "area",
        # Geographic features
        "jalan", "gedung", "rumah", "taman", "kantor", "pasar", "mall", 
        "kota", "desa", "provinsi", "kabupaten", "negara", "benua",
        # Directions
        "utara", "selatan", "timur", "barat", "tengah", "pinggir",
        # Common Indonesian locations
        "jakarta", "bandung", "surabaya", "yogyakarta", "bali", "lombok",
        "jawa", "sumatra", "kalimantan", "sulawesi", "papua"
    }
    
    # Person-related entities
    person_keywords = {
        # Person types
        "orang", "nama", "manusia", "pria", "wanita", "lelaki", "perempuan", 
        "anak", "bayi", "remaja", "dewasa", "orangtua", "keluarga",
        # Titles and roles
        "bapak", "ibu", "tuan", "nyonya", "nona", "pak", "bu", "mas", "mbak",
        "professor", "dokter", "guru", "siswa", "mahasiswa", "dosen",
        "direktur", "manajer", "karyawan", "pegawai", "pekerja",
        "presiden", "menteri", "gubernur", "bupati", "walikota"
    }
    
    # Organization-related entities
    org_keywords = {
        # Organization types
        "perusahaan", "organisasi", "lembaga", "badan", "institusi", "asosiasi",
        "yayasan", "komite", "komunitas", "kelompok", "tim", "partai",
        # Indonesian business forms
        "pt", "cv", "persero", "tbk", "firma", "koperasi",
        # Educational/governmental institutions
        "sekolah", "universitas", "perguruan", "tinggi", "akademi", "institut",
        "kementerian", "departemen", "dinas", "kantor", "kedutaan",
        # Health/public institutions
        "rumah sakit", "puskesmas", "klinik", "bank", "hotel", "restoran"
    }
    
    # Check which category the entity belongs to
    for keyword in time_keywords:
        if keyword in entity_lower:
            return "TIME"
    
    for keyword in location_keywords:
        if keyword in entity_lower:
            return "LOCATION"
    
    for keyword in person_keywords:
        if keyword in entity_lower:
            return "PERSON"
    
    for keyword in org_keywords:
        if keyword in entity_lower:
            return "ORGANIZATION"
    
    # Try WordNet-based categorization as fallback
    try:
        import nltk
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        
        # Check if entity exists in WordNet
        synsets = wn.synsets(entity, lang='ind')
        if not synsets:
            synsets = wn.synsets(entity)  # Fallback to English
        
        if synsets:
            # Get hypernyms to determine category
            for synset in synsets[:2]:  # Limit to first few synsets
                hypernym_paths = synset.hypernym_paths()
                for path in hypernym_paths:
                    for hypernym in path:
                        name = hypernym.name().lower()
                        if 'time' in name or 'date' in name or 'period' in name:
                            return "TIME"
                        if 'location' in name or 'place' in name or 'area' in name:
                            return "LOCATION"
                        if 'person' in name or 'human' in name or 'people' in name:
                            return "PERSON"
                        if 'organization' in name or 'institution' in name or 'company' in name:
                            return "ORGANIZATION"
    except:
        # If WordNet lookup fails, continue to fallback
        pass
    
    # Default category
    return "OTHER"
