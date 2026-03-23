import pandas as pd
import networkx as nx

def build_disease_drug_graph(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path, low_memory=False)
    
    # 1. Lowercase types AND names for perfect matching later
    df['x_type'] = df['x_type'].astype(str).str.lower()
    df['y_type'] = df['y_type'].astype(str).str.lower()
    df['x_name'] = df['x_name'].astype(str).str.lower()
    df['y_name'] = df['y_name'].astype(str).str.lower()
    
    # 2. Filter for only disease-drug OR drug-disease relationships
    print("Filtering for drugs and diseases...")
    mask = ((df['x_type'] == 'disease') & (df['y_type'] == 'drug')) | \
           ((df['x_type'] == 'drug') & (df['y_type'] == 'disease'))
    
    disease_drug_df = df[mask]
    
    # 3. Create the NetworkX graph directly from the DataFrame
    print("Constructing NetworkX graph...")
    G = nx.from_pandas_edgelist(
        disease_drug_df,
        source='x_name',
        target='y_name',
        edge_attr=['relation', 'display_relation'], 
        create_using=nx.Graph()                     
    )
    
    # 4. THE FIX: Assign the node types back onto the NetworkX nodes
    print("Assigning node types...")
    for _, row in disease_drug_df.iterrows():
        G.nodes[row['x_name']]['type'] = row['x_type']
        G.nodes[row['y_name']]['type'] = row['y_type']
    
    return G

# Execute the build
file_path = r"D:\Downloads\kg.csv" # Using 'r' for raw string to handle Windows backslashes
disease_drug_graph = build_disease_drug_graph(file_path)

print(f"\nSuccess! Graph built with {disease_drug_graph.number_of_nodes()} nodes and {disease_drug_graph.number_of_edges()} edges.")
nx.write_gexf(disease_drug_graph, "graph.gexf")