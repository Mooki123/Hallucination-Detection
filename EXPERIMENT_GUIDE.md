# Research Experiment Guide: LLM Hallucination Detection via Knowledge Graph Verification

A step-by-step guide for generating publishable results for your research paper.

---

## Phase 1: Environment Setup

### Step 1.1 — Install Dependencies
```bash
pip install networkx pandas openai rapidfuzz requests
```

### Step 1.2 — Set API Key
```powershell
$env:GROQ_API_KEY="your-groq-api-key"
```

### Step 1.3 — Verify Graph Files Exist
Ensure these files are present in the project root:
- `merged_graph.gexf` — the unified multi-source knowledge graph

If missing, rebuild the graph pipeline (see Phase 0 in Appendix).

---

## Phase 2: Preparing a Clean Experiment Run

### Step 2.1 — Clear Previous Results

> [!IMPORTANT]
> The CSV is in **append mode**. Before your final experiment run, delete or rename the existing file to avoid mixing test data with final results.

```powershell
# Backup old results
Move-Item "results/experiment_results.csv" "results/experiment_results_backup.csv" -ErrorAction SilentlyContinue
```

### Step 2.2 — Set the Sample Size in `main.py`

For a research paper, you need statistical significance. Modify line ~189 in `main.py`:

```python
# Recommended: 100 questions minimum for a research paper
# 200–300 is ideal for robust per-category analysis
test_questions = generate_baseline_questions(G, 200)
```

| Sample Size | Suitability |
|-------------|------------|
| 10–30       | Debugging / pilot run |
| 100         | Minimum for a paper (sparse per-category data) |
| **200–300** | **Recommended for research paper** |
| 500+        | Strong statistical power, but high API cost & time |

### Step 2.3 — Set a Random Seed for Reproducibility

> [!IMPORTANT]
> Research papers require reproducibility. Add this line **before** `generate_baseline_questions()`:

```python
random.seed(42)  # Fixed seed for reproducible question generation
test_questions = generate_baseline_questions(G, 200)
```

### Step 2.4 — Estimate Run Time and API Cost

Each question makes **2 LLM API calls** (one for answering, one for triple extraction) with **4-second delays** between calls. Estimate:

```
Time ≈ num_questions × (8 seconds delay + ~2–4s API latency) ≈ num_questions × 12 seconds
```

| Questions | Estimated Time |
|-----------|---------------|
| 100       | ~20 minutes |
| 200       | ~40 minutes |
| 300       | ~60 minutes |

---

## Phase 3: Running the Experiment

### Step 3.1 — Execute

```bash
cd "d:\hallucination project"
python code/main.py
```

Monitor the terminal output. You'll see real-time progress like:
```
[1/200] Processing Question: ...
    🔗 Fuzzy matched disease: 'high blood pressure' → 'hypertension'
[2/200] Processing Question: ...
```

### Step 3.2 — Verify Output

After completion, check that the CSV was created:
```powershell
python -c "import pandas as pd; df = pd.read_csv('results/experiment_results.csv'); print(f'Total claims: {len(df)}'); print(df['Verification_Status'].value_counts())"
```

This gives you an instant summary like:
```
Total claims: 487
Verified             198
Hallucination        142
Unverifiable          97
Relation Mismatch     50
```

---

## Phase 4: Analyzing Results for Your Paper

### Step 4.1 — Generate Summary Statistics

Create and run this script as `code/analyze_results.py`:

```python
import pandas as pd

df = pd.read_csv("results/experiment_results.csv")

print("=" * 60)
print("        EXPERIMENT SUMMARY STATISTICS")
print("=" * 60)

# 1. Overall Distribution
print("\n--- Overall Verification Distribution ---")
counts = df['Verification_Status'].value_counts()
percentages = df['Verification_Status'].value_counts(normalize=True) * 100
summary = pd.DataFrame({'Count': counts, 'Percentage (%)': percentages.round(2)})
print(summary)

# 2. Key Metrics (excluding Unverifiable for accuracy calculation)
resolvable = df[df['Verification_Status'] != 'Unverifiable']
total_resolvable = len(resolvable)

if total_resolvable > 0:
    verified = len(resolvable[resolvable['Verification_Status'] == 'Verified'])
    hallucinated = len(resolvable[resolvable['Verification_Status'] == 'Hallucination'])
    mismatched = len(resolvable[resolvable['Verification_Status'] == 'Relation Mismatch'])

    print(f"\n--- Key Metrics (Resolvable Claims Only: {total_resolvable}) ---")
    print(f"Factual Accuracy Rate:    {verified/total_resolvable*100:.1f}%")
    print(f"Hallucination Rate:       {hallucinated/total_resolvable*100:.1f}%")
    print(f"Relation Mismatch Rate:   {mismatched/total_resolvable*100:.1f}%")

# 3. Coverage
print(f"\n--- Entity Resolution Coverage ---")
print(f"Total extracted claims:   {len(df)}")
print(f"Successfully resolved:    {total_resolvable} ({total_resolvable/len(df)*100:.1f}%)")
print(f"Unverifiable:             {len(df) - total_resolvable} ({(len(df)-total_resolvable)/len(df)*100:.1f}%)")

# 4. Unique Questions
print(f"\n--- Dataset Summary ---")
print(f"Unique questions posed:   {df['Question'].nunique()}")
print(f"Total claims extracted:   {len(df)}")
print(f"Avg claims per question:  {len(df)/df['Question'].nunique():.1f}")
```

Run it:
```bash
python code/analyze_results.py
```

### Step 4.2 — Metrics to Report in Your Paper

Your paper should include these key metrics:

| Metric | Formula | What It Shows |
|--------|---------|--------------|
| **Hallucination Rate** | `Hallucination / (Verified + Hallucination + Mismatch)` | Core finding — how often the LLM fabricates |
| **Factual Accuracy** | `Verified / (Verified + Hallucination + Mismatch)` | How often the LLM is correct |
| **Relation Mismatch Rate** | `Mismatch / (Verified + Hallucination + Mismatch)` | Subtle errors (right entities, wrong relation) |
| **KG Coverage** | `Resolvable / Total Claims` | What proportion of claims your KG can verify |
| **Avg Claims per Question** | `Total Claims / Unique Questions` | Extraction richness |

### Step 4.3 — Generate a Publication-Ready Results Table

```python
# Add to analyze_results.py
print("\n--- LaTeX Table (copy-paste into your paper) ---")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{LLM Hallucination Detection Results}")
print(r"\begin{tabular}{|l|r|r|}")
print(r"\hline")
print(r"\textbf{Status} & \textbf{Count} & \textbf{\%} \\")
print(r"\hline")
for status, count in counts.items():
    pct = percentages[status]
    print(f"{status} & {count} & {pct:.1f}\\% \\\\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")
```

---

## Phase 5: Suggested Improvements for Stronger Results

### Improvement 1 — Test Multiple LLM Models (High Impact)

Run the same questions against 2–3 different models and compare hallucination rates. This is a **strong contribution** for a research paper.

In `config.py`, change `MODEL_ID` between runs:
```python
# Run 1
MODEL_ID = "llama-3.3-70b-versatile"
# Run 2
MODEL_ID = "gemma2-9b-it"
# Run 3
MODEL_ID = "deepseek-r1-distill-llama-70b"
```

Save results to separate files for each model run:
```python
# In main.py, modify save_path
save_path = f"results/experiment_results_{MODEL_ID.replace('/', '_')}.csv"
```

### Improvement 2 — Add Adversarial Questions (High Impact)

The current question generator only uses "forgiving" baseline questions. Add a **5th adversarial question type** that deliberately asks about non-existent relationships:

```python
elif question_type == 5:
    # Adversarial: Pair a disease with a RANDOM drug it has NO connection to
    disease = random.choice(diseases)
    connected_drugs = set(G.neighbors(disease))
    unconnected_drugs = [d for d in drugs if d not in connected_drugs]
    if unconnected_drugs:
        wrong_drug = random.choice(unconnected_drugs)
        templates = [
            f"Is {wrong_drug} an effective treatment for {disease}?",
            f"Can {wrong_drug} be prescribed for {disease}?",
        ]
        prompt = random.choice(templates)
        expected = f"Expected: NOT an indication. Triplet ['{disease}', 'not_indication', '{wrong_drug}']."
```

Also update `random.randint(1, 4)` → `random.randint(1, 5)`.

### Improvement 3 — Track Question Type in Results (Medium Impact)

Currently the question type (`Baseline Type 1`, etc.) is generated but **not saved** to the CSV. Add it to see which question types the LLM struggles with:

```python
# In the results append block, add:
"Question_Type": test_dict['type'],
```

### Improvement 4 — Use `graph_with_umls.gexf` for Better Coverage (Medium Impact)

The UMLS-enriched graph has additional metadata. You can still use fuzzy matching only but load the richer graph:
```python
filepath = "graph_with_umls.gexf"
```

### Improvement 5 — Add a Confidence Column (Low Impact, Nice-to-Have)

Modify `fuzzy_match_node` to return the match score alongside the match:

```python
def fuzzy_match_node(entity, all_nodes, score_cutoff=80.0):
    entity = str(entity).strip().lower()
    match = process.extractOne(entity, all_nodes, scorer=fuzz.token_sort_ratio, score_cutoff=score_cutoff)
    if match:
        return match[0], match[1]  # (node_name, confidence_score)
    return None, 0.0
```

This lets you report average entity resolution confidence in your paper.

---

## Phase 0 (Appendix): Rebuilding the Knowledge Graph from Scratch

Only needed if `merged_graph.gexf` is missing or you want to regenerate it.

```bash
# Step 0.1: Build base graph from PrimeKG (requires kg.csv)
python code/disease_drug_graph.py
# Output: graph.gexf

# Step 0.2: Build Hetionet subgraph (requires hetionet-v1.0.json in hetionet/)
python code/hetionet_graph.py
# Output: hetionet_graph.gexf

# Step 0.3: Fetch Wikidata drug-disease relationships
python wikidata.py
# Output: wikidata_drugs_diseases.csv

# Step 0.4: Merge all sources into unified graph
python merge.py
# Output: merged_graph.gexf

# (Optional) Step 0.5: Enrich with UMLS CUIs
python umls_graph.py
# Output: graph_with_umls.gexf
```

---

## Quick Checklist Before Final Run

- [ ] Cleared/renamed old `experiment_results.csv`
- [ ] Set `random.seed(42)` for reproducibility
- [ ] Set sample size to 200+ in `generate_baseline_questions()`
- [ ] Verified API key is set (`$env:GROQ_API_KEY`)
- [ ] `merged_graph.gexf` exists and loads correctly
- [ ] Sufficient time allocated (200 questions ≈ 40 min)
- [ ] (Optional) Added adversarial question type
- [ ] (Optional) Added question type tracking to results

---

## Expected Paper Sections You Can Fill from This Data

| Paper Section | Data Source |
|--------------|------------|
| **Abstract** | Overall hallucination rate + accuracy rate |
| **Methodology** | Question generation strategy, triple extraction pipeline, fuzzy matching approach |
| **Results Table 1** | Verification status distribution (from `analyze_results.py`) |
| **Results Table 2** | Per-question-type breakdown (if Improvement 3 applied) |
| **Results Table 3** | Cross-model comparison (if Improvement 1 applied) |
| **Discussion** | Analysis of Unverifiable claims, Relation Mismatches, KG coverage gaps |
| **Limitations** | KG coverage, entity resolution accuracy, single-hop verification |
