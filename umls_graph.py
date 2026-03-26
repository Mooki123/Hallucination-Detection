import networkx as nx
import spacy
from scispacy.linking import EntityLinker

def assign_umls_ids_to_graph(input_filepath, output_filepath):
    print("Loading the graph...")
    G = nx.read_gexf(input_filepath)
    
    print("Loading SciSpacy model and UMLS linker (this may take a minute)...")
    # Load the core biomedical model
    nlp = spacy.load("en_core_sci_sm")
    
    # Add the UMLS linker to the pipeline
    # resolve_abbreviations=True helps if your nodes have acronyms
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    
    # Get the linker component so we can access the knowledge base later if needed
    linker = nlp.get_pipe("scispacy_linker")

    print(f"Processing {G.number_of_nodes()} nodes...")
    
    mapped_count = 0
    unmapped_count = 0

    # Iterate through all nodes in the graph
    for node_name in G.nodes():
        # Clean the string just in case
        clean_name = str(node_name).strip().lower()
        
        # Pass the node name through the NLP pipeline
        doc = nlp(clean_name)
        
        best_cui = "UNKNOWN"
        
        # Check if SciSpacy recognized any entities in the string
        if doc.ents:
            # We take the first recognized entity (usually the whole string for a node name)
            entity = doc.ents[0]
            
            # kb_ents is a list of tuples: (CUI, match_score) sorted by score
            if entity._.kb_ents:
                # Grab the CUI with the highest confidence score
                best_cui, best_score = entity._.kb_ents[0]
        
        # Assign the found CUI (or "UNKNOWN") as a new attribute to the node
        G.nodes[node_name]['umls_id'] = best_cui
        
        if best_cui != "UNKNOWN":
            mapped_count += 1
        else:
            unmapped_count += 1

    print(f"\nFinished mapping!")
    print(f"Nodes successfully linked to UMLS: {mapped_count}")
    print(f"Nodes left unmapped (UNKNOWN): {unmapped_count}")

    print(f"\nSaving enriched graph to {output_filepath}...")
    nx.write_gexf(G, output_filepath)
    print("Done.")

# --- Execution ---
if __name__ == "__main__":
    input_graph_file = "merged_graph.gexf"            # The file you generated previously
    output_graph_file = "graph_with_umls.gexf" # The new enriched file
    
    assign_umls_ids_to_graph(input_graph_file, output_graph_file)