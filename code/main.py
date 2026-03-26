# imports 
import warnings
warnings.filterwarnings("ignore")
from utils import ask_llm, extract_triples
import networkx as nx
import re
import ast
import time
import random
import pandas as pd
import os
from rapidfuzz import process, fuzz


# Load graph 
filepath = "merged_graph.gexf"

G = nx.read_gexf(filepath)
print(f"Graph successfully loaded!")
print(f"Total Nodes: {G.number_of_nodes()}")
print(f"Total Edges: {G.number_of_edges()}")

#functions 
def parse_triples_string(llm_output_string):
    """Extracts the list from the LLM output using multiple fallback strategies."""
    
    # Strategy 1: Look for <python> tags
    match = re.search(r'<python>(.*?)</python>', llm_output_string, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return ast.literal_eval(match.group(1).strip())
        except Exception as e:
            print(f"⚠️ Failed to parse from <python> tags: {e}")
    
    # Strategy 2: Look for ```python or ``` code blocks
    match = re.search(r'```(?:python)?\s*(\[.*?\])\s*```', llm_output_string, re.DOTALL)
    if match:
        try:
            return ast.literal_eval(match.group(1).strip())
        except Exception as e:
            print(f"⚠️ Failed to parse from code block: {e}")
    
    # Strategy 3: Look for any nested list pattern [[...], [...]]
    match = re.search(r'(\[\s*\[.*?\]\s*\])', llm_output_string, re.DOTALL)
    if match:
        try:
            return ast.literal_eval(match.group(1).strip())
        except Exception as e:
            print(f"⚠️ Failed to parse raw list pattern: {e}")
    
    return []



def fuzzy_match_node(entity, all_nodes, score_cutoff=80.0):
    entity = str(entity).strip().lower()
    
    # Use token_sort_ratio to prevent partial-word hallucinations
    match = process.extractOne(
        entity, 
        all_nodes, 
        scorer=fuzz.token_sort_ratio, 
        score_cutoff=score_cutoff
    )
    
    if match:
        return match[0] 
    return None

def verify_relation_fuzzy(disease, relation, drug, graph, all_nodes):
    """Verifies a (disease, relation, drug) triple against the knowledge graph using fuzzy matching."""
    disease_input = str(disease).strip().lower()
    relation_input = str(relation).strip().lower()
    drug_input = str(drug).strip().lower()
    
    # --- Fuzzy Match Entities ---
    matched_disease = fuzzy_match_node(disease_input, all_nodes)
    matched_drug = fuzzy_match_node(drug_input, all_nodes)

    if matched_disease and matched_disease != disease_input:
        print(f"    🔗 Fuzzy matched disease: '{disease_input}' → '{matched_disease}'")
    if matched_drug and matched_drug != drug_input:
        print(f"    🔗 Fuzzy matched drug: '{drug_input}' → '{matched_drug}'")

    # --- Verification Logic ---
    if not matched_disease:
        return {"status": "Unverifiable", "reason": f"Disease '{disease_input}' not found in dataset.", "matched_disease": None, "matched_drug": None}
        
    if not matched_drug:
        return {"status": "Unverifiable", "reason": f"Drug '{drug_input}' not found in dataset.", "matched_disease": matched_disease, "matched_drug": None}
        
    if graph.has_edge(matched_disease, matched_drug):
        actual_kg_relation = str(graph[matched_disease][matched_drug].get('relation', 'unknown')).lower()
        
        if relation_input == "not_indication":
            return {"status": "Hallucination", "reason": f"LLM claims not an indication, but KG shows '{actual_kg_relation}'.", "matched_disease": matched_disease, "matched_drug": matched_drug}
            
        if relation_input == actual_kg_relation:
            return {"status": "Verified", "reason": "Correct relation.", "matched_disease": matched_disease, "matched_drug": matched_drug}
        else:
            return {"status": "Relation Mismatch", "reason": f"Expected '{actual_kg_relation}', got '{relation_input}'.", "matched_disease": matched_disease, "matched_drug": matched_drug}
    else:
        if relation_input == "not_indication":
            return {"status": "Verified", "reason": "Correctly identified lack of indication.", "matched_disease": matched_disease, "matched_drug": matched_drug}
        else:
            return {"status": "Hallucination", "reason": f"No edge between '{matched_disease}' and '{matched_drug}'.", "matched_disease": matched_disease, "matched_drug": matched_drug}


# generate questions 

def generate_baseline_questions(G, num_prompts=6):
    """
    Generates medical prompts across 5 question types:
    Types 1-4: Baseline knowledge questions (forgiving).
    Type 5: Adversarial false-claim questions (deliberately wrong drug-disease pairing).
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

    # Build adjacency set for fast adversarial pair generation
    connected_pairs = set((u, v) for u, v, _ in valid_indication_edges)
    connected_pairs.update((v, u) for u, v, _ in valid_indication_edges)

    prompts = []
    
    for _ in range(num_prompts):
        question_type = random.randint(1, 5)
        
        if question_type == 1:
            # Simple Disease Query — ask for a treatment for a given disease
            disease = random.choice(diseases)
            templates = [
                f"What is a common medication used to treat {disease}?",
                f"If a patient has {disease}, what drug is usually prescribed?",
                f"Can you name a standard treatment for {disease}?",
                f"A patient is diagnosed with {disease}. What medication would a physician typically consider?",
                f"What pharmacological treatment is indicated for {disease}?",
            ]
            prompt = random.choice(templates)
            expected = f"Expected: Valid drugs connected to '{disease}'."
            
        elif question_type == 2:
            # Simple Drug Query — ask what a drug is used for
            drug = random.choice(drugs)
            templates = [
                f"What medical condition is {drug} typically used to treat?",
                f"For what disease would a doctor usually prescribe {drug}?",
                f"What is the main use for the medication {drug}?",
                f"Which disease or condition is {drug} approved to treat?",
                f"What medical indication does {drug} have?",
            ]
            prompt = random.choice(templates)
            expected = f"Expected: Valid diseases connected to '{drug}'."
            
        elif question_type == 3:
            # Direct Confirmation — ask if a REAL known edge is valid
            disease, drug, _ = random.choice(valid_indication_edges)
            templates = [
                f"Is {drug} commonly prescribed to treat {disease}?",
                f"Can {drug} be used for a patient with {disease}?",
                f"I saw that {drug} is a treatment for {disease}. Is this correct?",
                f"My doctor mentioned {drug} for {disease}. Is that a recognized use?",
                f"Would a physician consider {drug} as a first-line treatment for {disease}?",
            ]
            prompt = random.choice(templates)
            expected = f"Expected: Triplet ['{disease}', 'indication', '{drug}']."
            
        elif question_type == 4:
            # Alternatives — ask for treatments *other than* a known drug
            disease, drug, _ = random.choice(valid_indication_edges)
            templates = [
                f"Besides {drug}, what is another common medication for {disease}?",
                f"If {drug} doesn't work for {disease}, what else can be used?",
                f"Can you list a few other treatments for {disease} other than {drug}?",
                f"What alternative medications exist for {disease} if {drug} is contraindicated?",
                f"A patient with {disease} is intolerant to {drug}. What other options are available?",
            ]
            prompt = random.choice(templates)
            expected = f"Expected: OTHER valid drugs connected to '{disease}'."

        elif question_type == 5:
            # Adversarial False Claim — pair a disease with a RANDOM drug that has NO connection
            # This tests whether the LLM hallucinates a relationship that doesn't exist in the KG
            disease = random.choice(diseases)
            connected_drugs = {v for u, v in connected_pairs if u == disease}
            unconnected_drugs = [d for d in drugs if d not in connected_drugs]
            
            if not unconnected_drugs:
                # Fallback to type 1 if no unconnected drug found
                prompt = f"What is a common medication used to treat {disease}?"
                expected = f"Expected: Valid drugs connected to '{disease}'."
            else:
                wrong_drug = random.choice(unconnected_drugs)
                templates = [
                    f"I read that {wrong_drug} is used to treat {disease}. Is this accurate?",
                    f"Is {wrong_drug} an established treatment for {disease}?",
                    f"Can {wrong_drug} be prescribed for {disease}?",
                    f"A friend told me {wrong_drug} is good for {disease}. Should I ask my doctor about it?",
                    f"Is it true that {wrong_drug} is indicated for {disease}?",
                ]
                prompt = random.choice(templates)
                expected = f"Expected: NOT an indication. Triplet ['{disease}', 'not_indication', '{wrong_drug}']."

        prompts.append({
            "prompt": prompt,
            "expected": expected,
            "type": f"Baseline Type {question_type}" if question_type <= 4 else "Adversarial Type 5"
        })
        
    return prompts

# --- Execution ---
# Generate 100 simple baseline queries
random.seed(42) 
test_questions = generate_baseline_questions(G, 120)
# --- Execution ---
# Generate 100 formal clinical queries for your paper's dataset




experiment_results = []

# Extract graph nodes exactly ONCE before the loop starts
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
        print(f"  📝 LLM Answer: {answer[:300]}...")
        print(f"  📝 Raw Extraction Output: {raw_llm_output[:500]}...")
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
            if rel_cleaned in ["indication", "contraindication", "off label use", "treated by", "not indication"]:
                
                ## 4. Use fuzzy verifier
                verification_dict = verify_relation_fuzzy(disease, rel_cleaned, drug, G, graph_nodes_list)

                # Append to our results list, saving both status and reason
                experiment_results.append({
                    "Question": q,
                    "LLM_Answer": answer,
                    "Extracted_Disease": disease,
                    "Matched_Disease": verification_dict["matched_disease"], # NEW
                    "Relation": rel_cleaned,
                    "Extracted_Drug": drug,
                    "Matched_Drug": verification_dict["matched_drug"], # NEW
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
    save_path = "results/experiment_results_qwen3-32b.csv"

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
