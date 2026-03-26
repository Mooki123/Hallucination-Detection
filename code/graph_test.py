import networkx as nx

def load_saved_graph(gexf_path):
    """Loads the pre-built graph from a GEXF file."""
    print(f"Loading graph from {gexf_path}...")
    # NetworkX handles loading the nodes, edges, and their attributes
    return nx.read_gexf(gexf_path)

def query_graph(G, query_name, target_type):
    """
    Finds all neighbors of a specific node that match a target type.
    """
    # Ensure the input is lowercase to match your graph's data
    query_name = str(query_name).lower()
    
    # Check if the node actually exists to prevent errors
    if not G.has_node(query_name):
        return []
    
    # Get neighbors and filter by the 'type' attribute 
    # (using .get() safely handles missing attributes)
    results = [
        neighbor for neighbor in G.neighbors(query_name) 
        if G.nodes[neighbor].get('type') == target_type
    ]
    
    return results

# ==========================================
# Execution
# ==========================================

# 1. Load the graph you built previously
gexf_file_path = "graph.gexf" 
G = load_saved_graph(gexf_file_path)

print(f"Graph loaded! Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# 2. Query: Find all drugs for a specific disease
target_disease = "chronic obstructive pulmonary disease"  # Change this to whatever disease you want
drugs_found = query_graph(G, target_disease, target_type="drug")

print(f"\n--- Drugs associated with '{target_disease}' ---")
if drugs_found:
    for drug in drugs_found:
        print(f"- {drug}")
else:
    print("No drugs found or disease not in graph.")

# 3. Query: Find all diseases for a specific drug
target_drug = "ibuprofen"  # Change this to whatever drug you want
diseases_found = query_graph(G, target_drug, target_type="disease")

print(f"\n--- Diseases associated with '{target_drug}' ---")
if diseases_found:
    for disease in diseases_found:
        print(f"- {disease}")
else:
    print("No diseases found or drug not in graph.")