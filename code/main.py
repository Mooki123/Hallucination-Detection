# imports 
import warnings
# Mute annoying ScispaCy and Sklearn version warnings in the terminal
warnings.filterwarnings("ignore")
from utils import ask_llm, extract_triples
import networkx as nx
import pickle
from openai import OpenAI
import re
import ast
import json
import time
import random
import pandas as pd
import os
from rapidfuzz import process, fuzz


# Load graph 
filepath = "graph.gexf"


G = nx.read_gexf(filepath)
print(f"Graph successfully loaded!")
print(f"Total Nodes: {G.number_of_nodes()}")
print(f"Total Edges: {G.number_of_edges()}")


#functions 
def parse_triples_string(llm_output_string):
    """Extracts the list from the <python> tags and safely evaluates it."""
    match = re.search(r'<python>(.*?)</python>', llm_output_string, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            # ast.literal_eval safely converts the string representation of a list into a real list
            return ast.literal_eval(match.group(1).strip())
        except Exception as e:
            print(f"⚠️ Failed to parse list: {e}")
            return []
    return []

import difflib

from rapidfuzz import process, fuzz

from rapidfuzz import process, fuzz

def fuzzy_match_node(entity, all_nodes, score_cutoff=80.0):
    entity = str(entity).strip().lower()
    
    # CHANGED: swapped WRatio for token_sort_ratio to prevent partial-word hallucinations
    match = process.extractOne(
        entity, 
        all_nodes, 
        scorer=fuzz.token_sort_ratio, 
        score_cutoff=score_cutoff
    )
    
    if match:
        return match[0] 
    return None

# Notice we added `all_nodes` as a required parameter here
def verify_relation_with_fuzzy(disease, relation, drug, graph, all_nodes):
    disease_input = str(disease).strip().lower()
    relation_input = str(relation).strip().lower()
    drug_input = str(drug).strip().lower()
    
    # Fuzzy match using the pre-calculated list
    matched_disease = fuzzy_match_node(disease_input, all_nodes)
    if not matched_disease:
        # Changed from Hallucination to Unverifiable
        return {"status": "Unverifiable", "reason": f"Disease '{disease_input}' not found in dataset."}
        
    matched_drug = fuzzy_match_node(drug_input, all_nodes)
    if not matched_drug:
        # Changed from Hallucination to Unverifiable
        return {"status": "Unverifiable", "reason": f"Drug '{drug_input}' not found in dataset."}
        
    if graph.has_edge(matched_disease, matched_drug):
        actual_kg_relation = str(graph[matched_disease][matched_drug].get('relation', 'unknown')).lower()
        if relation_input == actual_kg_relation:
            return {"status": "Verified", "reason": "Correct relation."}
        else:
            return {"status": "Relation Mismatch", "reason": f"Expected '{actual_kg_relation}', got '{relation_input}'."}
    else:
        return {"status": "Hallucination", "reason": f"No edge between '{matched_disease}' and '{matched_drug}'."}

# --- The Batch Loop ---

# 1. Extract the nodes EXACTLY ONCE before the loop starts
print("Extracting graph nodes...")
graph_nodes_list = list(G.nodes())



    

# generate questions 
import random
import pandas as pd

import random

import random
import pandas as pd

import random
import pandas as pd

import random

def generate_baseline_questions(G, num_prompts=6):
    """
    Generates simple, forgiving, and straightforward medical prompts.
    Focuses on baseline knowledge rather than adversarial traps.
    """
    # 1. Safely gather nodes by type
    diseases = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'disease']
    drugs = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'drug']
    
    # 2. Safely gather edges that ONLY connect a disease and a drug
    # This standardizes the order to (disease, drug) regardless of graph directionality
    valid_indication_edges = []
    for u, v, data in G.edges(data=True):
        u_type = G.nodes[u].get('type')
        v_type = G.nodes[v].get('type')
        
        if u_type == 'disease' and v_type == 'drug':
            valid_indication_edges.append((u, v, data))
        elif u_type == 'drug' and v_type == 'disease':
            valid_indication_edges.append((v, u, data)) # Standardize order
            
    # 3. Safety check to prevent IndexError
    if not diseases or not drugs or not valid_indication_edges:
        print("Warning: Graph is missing necessary nodes or edges.")
        return []

    prompts = []
    
    for _ in range(num_prompts):
        question_type = random.randint(1, 4)
        
        if question_type == 1:
            # Simple Disease Query
            disease = random.choice(diseases)
            templates = [
                f"What is a common medication used to treat {disease}?",
                f"If a patient has {disease}, what drug is usually prescribed?",
                f"Can you name a standard treatment for {disease}?"
            ]
            prompt = random.choice(templates)
            expected = f"Expected: Valid drugs connected to '{disease}'."
            
        elif question_type == 2:
            # Simple Drug Query
            drug = random.choice(drugs)
            templates = [
                f"What medical condition is {drug} typically used to treat?",
                f"For what disease would a doctor usually prescribe {drug}?",
                f"What is the main use for the medication {drug}?"
            ]
            prompt = random.choice(templates)
            expected = f"Expected: Valid diseases connected to '{drug}'."
            
        elif question_type == 3:
            # Direct Confirmation 
            disease, drug, _ = random.choice(valid_indication_edges)
            templates = [
                f"Is {drug} commonly prescribed to treat {disease}?",
                f"Can {drug} be used for a patient with {disease}?",
                f"I saw that {drug} is a treatment for {disease}. Is this correct?"
            ]
            prompt = random.choice(templates)
            expected = f"Expected: Triplet ['{disease}', 'indication', '{drug}']."
            
        elif question_type == 4:
            # Simple Alternatives 
            disease, drug, _ = random.choice(valid_indication_edges)
            templates = [
                f"Besides {drug}, what is another common medication for {disease}?",
                f"If {drug} doesn't work for {disease}, what else can be used?",
                f"Can you list a few other treatments for {disease} other than {drug}?"
            ]
            prompt = random.choice(templates)
            expected = f"Expected: OTHER valid drugs connected to '{disease}'."

        prompts.append({
            "prompt": prompt,
            "expected": expected,
            "type": f"Baseline Type {question_type}"
        })
        
    return prompts

# --- Execution ---
# Generate 100 simple baseline queries
test_questions = generate_baseline_questions(G, 4)
# --- Execution ---
# Generate 100 formal clinical queries for your paper's dataset




import os
import pandas as pd
import time

# Assuming eval_dataset_df and test_questions are already generated from your prompt generator

experiment_results = []

# 1. CRITICAL FIX: Extract graph nodes exactly ONCE before the loop starts
print("Extracting graph nodes for fuzzy matching...")
graph_nodes_list = list(G.nodes())

print("Starting LLM queries and verification...\n")
for i, test_dict in enumerate(test_questions):
    q = test_dict['prompt']
    print(f"[{i+1}/{len(test_questions)}] Processing Question: {q}")

    # 1. Get LLM Answer
    try:
        answer = ask_llm(q) 
    except Exception as e:
        print(f"  ⚠️ Error querying LLM: {e}")
        continue
    
    time.sleep(4) # Respect API rate limits

    # 2. Extract Claims
    try:
        raw_llm_output = extract_triples(answer) 
    except Exception as e:
        print(f"  ⚠️ Error extracting triples: {e}")
        continue
        
    time.sleep(4)

    # Convert the raw text string into a real Python list
    extracted_triples = parse_triples_string(raw_llm_output)
    
    if not extracted_triples:
        print("  ⚠️ No valid triples extracted for this question.")
        continue

    # 3. Verify Claims and Store Data
    for triple in extracted_triples:
        # Safety check: ensure the LLM returned exactly 3 items
        if len(triple) == 3:
            disease = str(triple[0]).strip()
            relation = str(triple[1]).strip()
            drug = str(triple[2]).strip() # Swapped from symptom to drug

            # Clean relation to handle minor LLM formatting quirks
            rel_cleaned = relation.lower().replace("_", " ").replace("-", " ")

            # Check against our PrimeKG allowed relationships
            if rel_cleaned in ["indication", "contraindication", "off label use", "treated by"]:
                
                # 4. FIX: Use the new fuzzy verifier and pass in the pre-calculated graph_nodes_list
                verification_dict = verify_relation_with_fuzzy(disease, rel_cleaned, drug, G, graph_nodes_list)

                # Append to our results list, saving both status and reason
                experiment_results.append({
                    "Question": q,
                    "LLM_Answer": answer,
                    "Extracted_Disease": disease,
                    "Relation": rel_cleaned,
                    "Extracted_Drug": drug,
                    "Verification_Status": verification_dict["status"],
                    "Verification_Reason": verification_dict["reason"]
                })
            else:
                print(f"  ⏭️ Skipping relation '{relation}': Not supported by current graph structure.")
        else:
            print(f"  ⚠️ Skipping malformed triple: {triple}")

# ==========================================
# 6. SAVE AND DISPLAY RESULTS (APPEND MODE)
# ==========================================
results_df = pd.DataFrame(experiment_results)

if not results_df.empty:
    # 1. Create a local 'results' folder automatically if it doesn't exist
    os.makedirs("results", exist_ok=True)
    save_path = "results/experiment_results.csv"

    # 2. Check if the file already exists so we know whether to write the header
    file_exists = os.path.isfile(save_path)

    # 3. Append the new data to the CSV
    results_df.to_csv(
        save_path, 
        mode='a',              # 'a' stands for append
        index=False, 
        header=not file_exists # Only write the header if the file is brand new
    )

    print("\n" + "="*50)
    print("              EXPERIMENT COMPLETE")
    print("="*50)
    print(f"New claims extracted and verified in this run: {len(results_df)}")
    print(f"Results successfully appended to: {os.path.abspath(save_path)}\n")

    # 4. Print a clean, readable table in the VS Code terminal
    print(results_df[['Extracted_Disease', 'Extracted_Drug', 'Verification_Status']].head().to_string(index=False))
else:
    print("\n⚠️ Experiment complete, but no valid claims were extracted. Nothing to save.")
