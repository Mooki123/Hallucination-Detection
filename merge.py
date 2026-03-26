import networkx as nx
import pandas as pd
from rapidfuzz import process, fuzz

def merge_wikidata_into_graph(gexf_path, wikidata_csv_path, output_gexf_path, threshold=90):
    print(f"1. Loading existing graph from {gexf_path}...")
    G = nx.read_gexf(gexf_path)
    
    existing_drugs = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'drug']
    existing_diseases = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'disease']
    
    print(f"   Found {len(existing_drugs)} drugs and {len(existing_diseases)} diseases in base graph.")

    print(f"2. Loading new Wikidata edges from {wikidata_csv_path}...")
    col_names = ['x_name', 'x_type', 'y_name', 'y_type', 'relation']
    wiki_df = pd.read_csv(wikidata_csv_path, names=col_names, header=None)
    
    drug_cache = {}
    disease_cache = {}

    def get_best_match(name, candidates, cache):
        name = str(name).lower()
        if name in cache:
            return cache[name]
            
        match = process.extractOne(name, candidates, scorer=fuzz.WRatio)
        
        if match and match[1] >= threshold:
            cache[name] = match[0]
        else:
            cache[name] = name
            candidates.append(name) 
            
        return cache[name]

    print(f"3. Fuzzy matching and injecting {len(wiki_df)} new edges...")
    
    new_nodes_added = 0
    new_edges_added = 0
    edges_updated = 0
    
    for _, row in wiki_df.iterrows():
        x_type = str(row['x_type']).strip().lower()
        y_type = str(row['y_type']).strip().lower()
        
        # 1. Assign drug and disease correctly regardless of column order
        if x_type == 'drug' and y_type == 'disease':
            drug_name, disease_name = row['x_name'], row['y_name']
        elif x_type == 'disease' and y_type == 'drug':
            drug_name, disease_name = row['y_name'], row['x_name']
        else:
            continue # Skip malformed rows
            
        mapped_drug = get_best_match(drug_name, existing_drugs, drug_cache)
        mapped_disease = get_best_match(disease_name, existing_diseases, disease_cache)
        
        # 2. Add missing nodes
        if not G.has_node(mapped_drug):
            G.add_node(mapped_drug, type='drug', source='wikidata_new')
            new_nodes_added += 1
            
        if not G.has_node(mapped_disease):
            G.add_node(mapped_disease, type='disease', source='wikidata_new')
            new_nodes_added += 1
            
        # 3. Standardize the relation to 'indication'
        raw_relation = str(row['relation']).strip().lower()
        if raw_relation in ['treated_by', 'treats']:
            standardized_relation = 'indication'
        else:
            standardized_relation = raw_relation

        # 4. Handle Edge Addition (No Multigraphs!)
        if not G.has_edge(mapped_drug, mapped_disease):
            # It's a brand new edge, add it with the standardized relation
            G.add_edge(mapped_drug, mapped_disease, relation=standardized_relation, source='wikidata')
            new_edges_added += 1
        else:
            # The edge already exists in the base graph. 
            # We don't overwrite the existing relation (e.g., if it's 'off-label-use'), 
            # but we update the source to show it was cross-verified by Wikidata.
            current_source = G[mapped_drug][mapped_disease].get('source', 'base_graph')
            if 'wikidata' not in current_source:
                G[mapped_drug][mapped_disease]['source'] = f"{current_source}_and_wikidata"
                edges_updated += 1

    print(f"4. Saving enriched graph to {output_gexf_path}...")
    nx.write_gexf(G, output_gexf_path)
    
    print("\n--- Merge Complete ---")
    print(f"Total Nodes: {G.number_of_nodes()} (+{new_nodes_added} genuinely new ones)")
    print(f"Total Edges: {G.number_of_edges()} (+{new_edges_added} new edges, {edges_updated} cross-verified edges)")
    
    return G
merge_wikidata_into_graph(r"graph.gexf",r"wikidata_drugs_diseases.csv",r"merged_graph.gexf")