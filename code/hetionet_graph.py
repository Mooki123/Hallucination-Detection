import json
import bz2
import networkx as nx

def build_hetionet_drug_disease_graph(filepath):
    print(f"Decompressing and loading Hetionet from {filepath}...")
    
    # 1. Open the bz2 compressed file and load the JSON
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    G = nx.Graph() 
    
    # 2. Map IDs to Names
    # Hetionet uses IDs (like 'DB01106') in its edges, but we want 
    # to search by name (like 'aspirin'). We build a dictionary to translate them.
    id_to_name = {}
    
    print("Extracting Compound and Disease nodes...")
    for node in data['nodes']:
        if node['kind'] in ['Compound', 'Disease']:
            node_name = str(node['name']).lower()
            node_id = str(node['identifier'])
            
            # Save to our translation dictionary
            id_to_name[node_id] = node_name
            
            # Add the node to NetworkX
            G.add_node(node_name, type=node['kind'].lower(), identifier=node_id)
            
    # 3. Add the Edges (Relationships)
    print("Connecting Drugs to Diseases...")
    for edge in data['edges']:
        source_type, source_id = edge['source_id']
        target_type, target_id = edge['target_id']
        
        # We only care about connections between Compounds and Diseases
        if source_type in ['Compound', 'Disease'] and target_type in ['Compound', 'Disease']:
            
            # Translate the IDs back into readable names
            source_name = id_to_name.get(str(source_id))
            target_name = id_to_name.get(str(target_id))
            
            # If both nodes exist in our filtered list, connect them!
            if source_name and target_name:
                G.add_edge(source_name, target_name, relation=edge['kind'].lower())
                
    return G

def query_disease_for_drugs(G, disease_name):
    """Finds all drugs that treat or palliate a specific disease."""
    disease_name = disease_name.lower()
    
    if not G.has_node(disease_name):
        return f"Disease '{disease_name}' not found."
    
    results = []
    for neighbor in G.neighbors(disease_name):
        if G.nodes[neighbor].get('type') == 'compound':
            # Get the specific relation (e.g., "treats" or "palliates")
            relation = G.edges[disease_name, neighbor]['relation']
            results.append(f"{neighbor} ({relation})")
            
    return results

# ==========================================
# Execution
# ==========================================

# Point this to wherever you downloaded the .bz2 file
file_path = r"D:\hallucination project\hetionet\hetionet-v1.0.json" 

hetionet_graph = build_hetionet_drug_disease_graph(file_path)

print(f"\nSuccess! Graph built with {hetionet_graph.number_of_nodes()} nodes and {hetionet_graph.number_of_edges()} edges.")

# Let's test it!
query = "asthma"
print(f"\nDrugs associated with {query}:")
drugs = query_disease_for_drugs(hetionet_graph, query)
for d in drugs:
    print(f"- {d}")
nx.write_gexf(hetionet_graph, "hetionet_graph.gexf")