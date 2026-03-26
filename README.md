# Detecting Biomedical Hallucinations in Large Language Models via Multi-Source Knowledge Graph Verification

---

## Abstract

Large Language Models (LLMs) have demonstrated remarkable fluency in answering clinical and biomedical queries; however, they remain susceptible to *hallucination* — the generation of factually incorrect or fabricated information. In the high-stakes domain of medicine, such errors can have severe real-world consequences. This work presents a structured, automated pipeline for detecting and quantifying biomedical hallucinations in LLMs using a curated, multi-source biomedical Knowledge Graph (KG) as a ground-truth verifier. The system generates diverse natural-language medical queries — spanning straightforward indication lookups, relational confirmations, alternative-treatment requests, and adversarial false-claim prompts — submits them to a target LLM via API, and uses a second LLM to extract structured (disease, relation, drug) triples from free-form responses. Each extracted triple is then verified against the KG using fuzzy entity resolution. The pipeline operates across four verification outcomes: *Verified*, *Hallucination*, *Relation Mismatch*, and *Unverifiable*. Experiments conducted on Llama-3.3-70B-Versatile and Qwen3-32B over 120 diverse prompts reveal systemic hallucination rates that vary meaningfully across question types, contributing empirical evidence for LLM reliability assessment in clinical contexts.

---

## 1. Problem Statement

### 1.1 Motivation

LLMs are increasingly integrated into clinical decision-support systems, patient-facing chatbots, and medical education tools. Unlike general-domain errors, biomedical hallucinations — such as recommending a drug that has no therapeutic relation to a stated disease, or misclassifying a contraindication as an indication — may directly mislead clinicians or patients.

### 1.2 Problem Definition

Given a natural-language biomedical question $Q$, an LLM $\mathcal{M}$ produces a free-text answer $A = \mathcal{M}(Q)$. The core challenge is:

> **Can the factual claims embedded in $A$ be automatically verified against a structured biomedical knowledge graph $\mathcal{G}$, and if so, at what rate does $\mathcal{M}$ hallucinate?**

Formally, a *claim* is represented as a triple $(e_d, r, e_p)$ where:
- $e_d \in \mathcal{V}_D$ — a disease entity
- $e_p \in \mathcal{V}_P$ — a pharmacological entity (drug)
- $r \in \{\text{indication, contraindication, off-label use, not\_indication}\}$ — a clinical relation

Hallucination is defined as: the LLM asserting $(e_d, r, e_p)$ when $\nexists$ a corresponding edge in $\mathcal{G}$ (or when the edge relation $r'$ in $\mathcal{G}$ conflicts with the asserted $r$).

### 1.3 Real-World Relevance

- **Patient Safety**: A hallucinated `indication` for a drug that is actually `contraindicated` could directly harm a patient acting on LLM advice.
- **Regulatory Compliance**: Healthcare AI systems increasingly face regulatory scrutiny (e.g., FDA AI/ML action plan); automated hallucination auditing provides a measurable compliance mechanism.
- **Benchmarking**: This pipeline provides a reproducible, domain-agnostic framework to compare hallucination rates across different LLM families and sizes.

---

## 2. Methodology

### 2.1 Overview

The pipeline comprises five stages:

```
[KG Construction] → [Question Generation] → [LLM Querying] → [Triple Extraction] → [KG Verification]
```

### 2.2 Knowledge Graph Construction

A unified biomedical knowledge graph $\mathcal{G}$ is constructed by merging three authoritative sources:

**Source 1 — PrimeKG** (`disease_drug_graph.py`): A large-scale precision medicine knowledge graph. Only `(disease, drug)` edges are retained. Node types are explicitly tagged as `disease` or `drug`. The graph is serialized to `graph.gexf`.

**Source 2 — Hetionet v1.0** (`hetionet_graph.py`): A heterogeneous biomedical network containing 47,031 nodes and 2.25M edges from 11 databases. The script filters for `Compound–Disease` edges only, maps internal DrugBank/MONDO identifiers to human-readable lowercase names, and serializes the subgraph.

**Source 3 — Wikidata** (`wikidata.py`): Drug–disease relationships are harvested via a SPARQL query against the Wikidata Query Service using properties `P2175` (*drug used for treatment*) and `P2176` (*medical treatment*). The fetched triples are normalized and deduplicated into `wikidata_drugs_diseases.csv`.

**Merging** (`merge.py`): The three sources are merged into a single unified graph `merged_graph.gexf`. During merging, fuzzy entity matching (RapidFuzz `WRatio` scorer, threshold ≥ 90) ensures that entities expressed differently across sources are canonicalized to a single node rather than duplicated. Relation labels `treated_by` and `treats` are standardized to `indication`. An edge that already exists in the base graph is not overwritten; instead, its `source` attribute is annotated to reflect cross-source verification.

**Optional UMLS Enrichment** (`umls_graph.py`): Node metadata is further enriched by mapping each entity to a UMLS Concept Unique Identifier (CUI) using SciSpacy (`en_core_sci_sm` model + `scispacy_linker` UMLS linker). This produces `graph_with_umls.gexf`.

### 2.3 Query Generation (`generate_baseline_questions` in `main.py`)

The system generates a stratified, randomized set of natural-language prompts across **five question types**, each targeting a distinct hallucination failure mode:

| Type | Category | Description | Hallucination Risk |
|------|----------|-------------|-------------------|
| 1 | Simple Disease Query | "What medication treats `{disease}`?" | Open-ended fabrication |
| 2 | Simple Drug Query | "What disease is `{drug}` used for?" | Entity confusion / off-target claim |
| 3 | Direct Confirmation | "Is `{drug}` prescribed for `{disease}`?" (TRUE edge) | Relation mis-labeling |
| 4 | Alternative Treatment | "Besides `{drug}`, what else treats `{disease}`?" | False recommendations |
| 5 | Adversarial False Claim | "Is `{wrong_drug}` used for `{disease}`?" (NO edge) | Confabulation of non-existent relations |

For Type 5, the adversarial drug $d'$ is explicitly selected such that $(e_d, *, d') \notin \mathcal{E}(\mathcal{G})$, testing whether the LLM correctly refuses to affirm a non-relationship. Five natural-language template variants are maintained per question type and sampled uniformly at random to mitigate template-specific biases. A fixed random seed (`seed=42`) is applied throughout for reproducibility.

### 2.4 LLM Querying (`ask_llm` in `utils.py`)

The target LLM $\mathcal{M}_1$ (e.g., `qwen/qwen3-32b` or `llama-3.3-70b-versatile`) is queried via the Groq API using an OpenAI-compatible client. Each prompt is prefixed with a concise medical assistant system instruction: *"Answer this query concisely."* No chain-of-thought or few-shot examples are injected at this stage, ensuring that responses reflect the model's raw clinical knowledge.

### 2.5 Triple Extraction (`extract_triples` in `utils.py`)

Free-text LLM responses are parsed into structured triples by a second, specialist LLM $\mathcal{M}_2$ (`openai/gpt-oss-120b` at temperature 0.0). The extraction prompt enforces:

1. **Entity normalization**: Brand-name drugs → generic INN names (e.g., "Advil" → "ibuprofen"; "Norvasc" → "amlodipine"). Disease names → formal MONDO-aligned terms (e.g., "high blood pressure" → "hypertension").
2. **Coreference resolution**: Duplicate and pronominally-referenced entities are collapsed.
3. **Strict relation vocabulary**: Only `{indication, contraindication, off-label use, not_indication}` are valid.
4. **Negation differentiation**: "not a standard treatment" → `not_indication`; "medically dangerous" → `contraindication`.
5. **Few-shot demonstrations**: Four exemplars covering all four relation types are provided in the prompt.

Output is structured as a Python list of lists wrapped in `<python>...</python>` tags. A multi-strategy parser (`parse_triples_string`) handles extraction robustness with three fallback strategies: tag-based regex, fenced code block regex, and raw nested-list pattern matching.

### 2.6 KG Verification (`verify_relation_fuzzy` in `main.py`)

Each extracted triple $(e_d^*, r^*, e_p^*)$ is verified against $\mathcal{G}$ in three steps:

**Step 1 — Fuzzy Entity Resolution**: The extracted entity strings are matched against all graph node names using RapidFuzz `token_sort_ratio` with a match-score cutoff of ≥ 80. This tolerates minor spelling variants, hyphenation differences, and LLM-induced surface form variations (e.g., "hodgkins lymphoma" → "hodgkin lymphoma").

**Step 2 — Edge Existence Check**: If both $e_d$ and $e_p$ resolve to graph nodes, the edge $(e_d, e_p) \in \mathcal{E}(\mathcal{G})$ is queried. If no edge exists, the system checks whether the asserted relation is `not_indication` (correct refusal) or some positive relation (hallucination).

**Step 3 — Relation Consistency Check**: If the edge exists, the asserted relation $r^*$ is compared to the stored edge attribute `relation`. A match yields *Verified*; a mismatch yields *Relation Mismatch*.

```
Verification Outcomes:
├── Verified          → edge ∈ G AND r* == r_KG
├── Hallucination     → edge ∉ G AND r* ≠ "not_indication"
│                     OR edge ∈ G AND r* == "not_indication"
├── Relation Mismatch → edge ∈ G AND r* ≠ r_KG
└── Unverifiable      → e_d or e_p not resolved in G
```

---

## 3. System Architecture

### 3.1 Component Interaction Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     KNOWLEDGE GRAPH LAYER                        │
│                                                                  │
│  PrimeKG (kg.csv)  ──┐                                          │
│  Hetionet v1.0.json ─┼──► merge.py ──► merged_graph.gexf (G)   │
│  Wikidata SPARQL   ──┘                      │                    │
│                         umls_graph.py ──► graph_with_umls.gexf  │
└─────────────────────────────────────────────┼────────────────────┘
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT PIPELINE (main.py)                 │
│                                                                  │
│  generate_baseline_questions(G) ─► [Q1, Q2, Q3, Q4, Q5] prompts│
│           │                                                      │
│           ▼                                                      │
│  ask_llm(prompt) ──► [Groq API / M1] ──► free-text answer A    │
│           │                                                      │
│           ▼                                                      │
│  extract_triples(A) ──► [Groq API / M2] ──► <python>[...]</python>│
│           │                                                      │
│           ▼                                                      │
│  parse_triples_string() ──► [(disease, rel, drug), ...]         │
│           │                                                      │
│           ▼                                                      │
│  verify_relation_fuzzy(triple, G) ──► {status, reason}          │
│           │                                                      │
│           ▼                                                      │
│  experiment_results[] ──► results/experiment_results_{model}.csv│
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow Summary

1. Raw KG sources → Graph construction scripts → `.gexf` serialized graphs
2. Loaded `merged_graph.gexf` → node/edge enumeration → stratified question pool
3. Question prompt → LLM $\mathcal{M}_1$ → natural-language answer
4. Answer text → LLM $\mathcal{M}_2$ (zero-temperature) → structured triples
5. Triples → fuzzy entity resolution → KG edge lookup → verification verdict
6. All verdicts → append-mode CSV export per model

---

## 4. Technologies Used

| Category | Tool / Library | Version / Notes |
|----------|---------------|-----------------|
| **Language** | Python 3.x | — |
| **Graph Processing** | NetworkX | GEXF read/write, edge/node manipulation |
| **Fuzzy Matching** | RapidFuzz | `fuzz.token_sort_ratio`, `fuzz.WRatio` |
| **LLM API** | OpenAI Python SDK | Used with Groq-compatible base URL |
| **LLM Inference** | Groq API | Low-latency inference endpoint |
| **Target LLM (M1)** | Qwen3-32B / Llama-3.3-70B-Versatile | Via `qwen/qwen3-32b`, `llama-3.3-70b-versatile` |
| **Extractor LLM (M2)** | GPT-OSS-120B | `openai/gpt-oss-120b`, temperature=0.0 |
| **Data Processing** | Pandas | CSV I/O, deduplication, result aggregation |
| **Biomedical NLP** | SciSpacy + SpaCy | `en_core_sci_sm` model for UMLS linking |
| **KG Source 1** | PrimeKG | CSV-format precision medicine KG |
| **KG Source 2** | Hetionet v1.0 | JSON format, 47K nodes / 2.25M edges |
| **KG Source 3** | Wikidata | SPARQL endpoint (P2175, P2176 properties) |
| **Ontologies** | MONDO, DrugBank, UMLS | Entity normalization standards |
| **Serialization** | GEXF | Graph Exchange XML Format |
| **Environment** | PowerShell / Windows | Development environment |

---

## 5. Dataset / Input

### 5.1 Knowledge Graph Sources

| Source | Format | Size (approx.) | Relation Types |
|--------|--------|----------------|----------------|
| PrimeKG | CSV (`kg.csv`) | ~5M+ rows, filtered for disease–drug | indication, contraindication, off-label use |
| Hetionet v1.0 | JSON (`hetionet-v1.0.json`) | 47,031 nodes, 2,250,197 edges | Treats, Palliates, etc. (mapped to `indication`) |
| Wikidata | SPARQL → CSV | Variable (~10K+ edges) | `treats`, `treated_by` (→ `indication`) |
| **Merged Graph** | GEXF | Combined, deduplicated | `indication`, `contraindication`, `off-label use` |

### 5.2 Preprocessing Steps

1. **Type Filtering**: Only `disease` and `drug`/`compound` node types are retained from each source; all other biological entity types (genes, pathways, etc.) are discarded.
2. **Lowercasing**: All node names and edge relations are converted to lowercase for consistent matching.
3. **Generic Name Normalization**: Brand-name drugs in LLM outputs are mapped to generic INN names by the extraction LLM.
4. **Fuzzy Deduplication** (merge): Entities within a Levenshtein-based similarity threshold of ≥ 90 are collapsed during the merge process using `RapidFuzz`.
5. **Relation Standardization**: `treats`/`treated_by` are unified to `indication`; the `off-label use` hyphenation is smoothed during verification (`off-label use` ↔ `off label use`).

### 5.3 Assumptions

- The merged KG represents a sufficient, though not exhaustive, gold standard. Edges absent from the KG do not necessarily represent false medical claims.
- All drug–disease edges in the KG are treated as semantically valid clinical relationships.
- Entity resolution errors (fuzzy match below the 80-point threshold) result in `Unverifiable` rather than `Hallucination` to avoid false positives.

---

## 6. Implementation Details

### 6.1 Key Functions

#### `generate_baseline_questions(G, num_prompts)` — `main.py`
Samples nodes and edges from $\mathcal{G}$ to construct prompts. For adversarial Type 5 questions, a connected-pairs set is pre-built as `set((u,v) for u,v,_ in valid_indication_edges)` enabling O(1) lookup when searching for unconnected drug candidates. A fallback to Type 1 is triggered if no unconnected drug is found (isolated disease node).

#### `parse_triples_string(llm_output_string)` — `main.py`
A three-strategy cascading parser:
1. `<python>...</python>` tag extraction via `re.search` + `ast.literal_eval`
2. Fenced code block (` ``` `) extraction
3. Raw nested list pattern `[\s*[...]\s*]` extraction

This design handles the full range of LLM output formatting inconsistencies without requiring a strict output format guarantee.

#### `fuzzy_match_node(entity, all_nodes, score_cutoff=80.0)` — `main.py`
Uses `fuzz.token_sort_ratio` (insensitive to word order) rather than simple ratio to handle cases like "hodgkin lymphoma" vs. "lymphoma hodgkin". The node list is extracted once before the main loop (`graph_nodes_list = list(G.nodes())`) to avoid repeated O(N) conversion inside the per-triple verification loop.

#### `verify_relation_fuzzy(disease, relation, drug, graph, all_nodes)` — `main.py`
Returns a structured dictionary `{status, reason, matched_disease, matched_drug}` rather than a plain string, enabling downstream logging of both the raw extracted entity and the canonical matched entity for error analysis.

#### `extract_triples(input_text)` — `utils.py`
Implements a four-step extraction pipeline encoded in the system prompt: (1) entity detection & normalization, (2) coreference resolution, (3) relation extraction, (4) relation categorization with strict vocabulary. Zero temperature (`temperature=0.0`) ensures deterministic triple extraction, critical for experimental reproducibility.

#### `merge_wikidata_into_graph(...)` — `merge.py`
Implements a cache-accelerated fuzzy merge: a `drug_cache` and `disease_cache` dictionary store previously computed fuzzy matches, reducing redundant `process.extractOne` calls for repeated entity names across the Wikidata CSV rows. New nodes introduced by Wikidata are tagged `source='wikidata_new'`; cross-verified edges receive a `source` attribute update (`base_graph_and_wikidata`).

### 6.2 Design Decisions

- **Append-mode CSV writing**: Results accumulate across multiple runs without overwriting, enabling incremental experimentation. A `file_exists` check ensures the CSV header is written only once.
- **Rate limiting**: 4-second `time.sleep()` calls between each LLM API call respect Groq API rate limits and prevent request throttling during long experiment runs.
- **Separation of answer and extraction LLMs**: Using two distinct models ($\mathcal{M}_1$ for answering, $\mathcal{M}_2$ for extraction) prevents the answering model's style from biasing the extraction process, and allows swapping the answering model freely without changing the extraction pipeline.
- **Relation cleaning**: Before verification, relation strings undergo `.lower().replace("_"," ").replace("-"," ")` normalization, harmonizing minor formatting variants (e.g., `off-label-use`, `off_label_use`, `off label use`).

---

## 7. Results / Output

### 7.1 Output Format

All verified claims are saved to `results/experiment_results_{model_name}.csv` with the following schema:

| Column | Description |
|--------|-------------|
| `Question` | Original natural-language prompt posed to the LLM |
| `LLM_Answer` | Full free-text response from the target LLM |
| `Extracted_Disease` | Disease entity as extracted by the extraction LLM |
| `Matched_Disease` | Canonical disease name after fuzzy resolution in KG |
| `Relation` | Normalized relation asserted by the LLM |
| `Extracted_Drug` | Drug entity as extracted by the extraction LLM |
| `Matched_Drug` | Canonical drug name after fuzzy resolution in KG |
| `Verification_Status` | One of: `Verified`, `Hallucination`, `Relation Mismatch`, `Unverifiable` |
| `Verification_Reason` | Human-readable explanation of the verdict |

### 7.2 Evaluation Metrics

| Metric | Formula |
|--------|---------|
| **Hallucination Rate** | $\frac{|\text{Hallucination}|}{|\text{Verified}| + |\text{Hallucination}| + |\text{Relation Mismatch}|}$ |
| **Factual Accuracy Rate** | $\frac{|\text{Verified}|}{|\text{Verified}| + |\text{Hallucination}| + |\text{Relation Mismatch}|}$ |
| **Relation Mismatch Rate** | $\frac{|\text{Relation Mismatch}|}{|\text{Verified}| + |\text{Hallucination}| + |\text{Relation Mismatch}|}$ |
| **KG Coverage (Resolvability)** | $\frac{|\text{Resolvable Claims}|}{|\text{Total Claims}|}$ |
| **Extraction Richness** | $\frac{|\text{Total Claims}|}{|\text{Unique Questions}|}$ |

> **Note**: The *Unverifiable* category is excluded from primary accuracy/hallucination rate calculations to avoid penalizing cases where KG coverage — rather than LLM correctness — is the limiting factor.

### 7.3 Experimental Results Summary

Results were collected for two LLMs (`Llama-3.3-70B-Versatile` and `Qwen3-32B`) over 120 diverse prompts (random seed = 42). The Llama-3.3-70B-Versatile CSV contains approximately 1,189 verified claim rows, providing statistically meaningful per-category distributions. Typical observed patterns include:

- **Verified** claims predominantly from well-known, high-frequency drug–disease pairs (e.g., rifampicin→tuberculosis, rufinamide→Lennox-Gastaut syndrome).
- **Hallucinations** frequently cluster around rare diseases, combination therapy contexts, and off-label uses where the LLM over-asserts `indication`.
- **Relation Mismatches** occur characteristically where the KG records `contraindication` but the LLM returns `not_indication` (a semantically adjacent but categorically distinct label).
- **Unverifiable** claims arise from LLM mention of compound names (e.g., "trimethoprim-sulfamethoxazole") that differ in hyphenation from the KG entry, or from disease sub-type specificity exceeding KG granularity.

---

## 8. How to Run

### 8.1 Prerequisites

- Python 3.8+
- A valid Groq API key with access to the required models

### 8.2 Step 1 — Install Dependencies

```bash
pip install networkx pandas openai rapidfuzz requests spacy
```

For UMLS enrichment (optional):
```bash
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
```

### 8.3 Step 2 — Set API Key

```powershell
# Windows PowerShell
$env:GROQ_API_KEY = "your-groq-api-key-here"
```

```bash
# Linux / macOS
export GROQ_API_KEY="your-groq-api-key-here"
```

### 8.4 Step 3 — Build the Knowledge Graph (First Time Only)

If `merged_graph.gexf` is not present, rebuild it from sources:

```bash
# Step 3a: Build base graph from PrimeKG (requires kg.csv in the path specified in disease_drug_graph.py)
python code/disease_drug_graph.py
# Output: graph.gexf

# Step 3b: Build Hetionet subgraph (requires hetionet/hetionet-v1.0.json)
python code/hetionet_graph.py
# Output: hetionet_graph.gexf

# Step 3c: Fetch Wikidata relationships
python wikidata.py
# Output: wikidata_drugs_diseases.csv

# Step 3d: Merge all sources
python merge.py
# Output: merged_graph.gexf

# (Optional) Step 3e: Add UMLS CUI annotations
python umls_graph.py
# Output: graph_with_umls.gexf
```

### 8.5 Step 4 — Configure the Experiment

Edit `code/config.py` to select the target LLM:

```python
MODEL_ID1 = "llama-3.3-70b-versatile"   # Target LLM (answering)
MODEL_ID2 = "openai/gpt-oss-120b"        # Extraction LLM
```

In `code/main.py`, set the number of questions and random seed:

```python
random.seed(42)                           # Reproducibility
test_questions = generate_baseline_questions(G, 200)  # ≥200 for paper-quality results
```

Update the output filename to reflect the model under test:
```python
save_path = "results/experiment_results_llama-3.3-70b-versatile.csv"
```

### 8.6 Step 5 — Run the Experiment

```bash
cd "d:\hallucination project"
python code/main.py
```

Monitor real-time progress in the terminal. Each question prints its index, the fuzzy-matched entities, and the verification verdict.

### 8.7 Step 6 — Analyze Results

```python
import pandas as pd

df = pd.read_csv("results/experiment_results_llama-3.3-70b-versatile.csv")
resolvable = df[df['Verification_Status'] != 'Unverifiable']

print(df['Verification_Status'].value_counts())
print(f"Hallucination Rate: {len(resolvable[resolvable['Verification_Status']=='Hallucination'])/len(resolvable)*100:.1f}%")
print(f"Factual Accuracy:   {len(resolvable[resolvable['Verification_Status']=='Verified'])/len(resolvable)*100:.1f}%")
```

---

## 9. Limitations

1. **Knowledge Graph Incompleteness**: The merged KG, while multi-source, does not exhaustively cover all valid drug–disease relationships. Valid LLM claims may be classified as `Unverifiable` due to KG gaps. This introduces false negatives in hallucination detection.

2. **Single-Hop Verification Only**: The pipeline verifies only direct disease–drug edges. It cannot validate multi-step clinical reasoning (e.g., "Drug A reduces biomarker B, which is associated with disease C").

3. **Fuzzy Matching Errors**: The 80-point score cutoff for entity resolution may incorrectly match semantically distinct entities with similar surface forms (false positives) or fail to match legitimate variants beyond the threshold (false negatives). The `token_sort_ratio` scorer mitigates word-order sensitivity but not semantic drift.

4. **Extraction LLM Dependence**: Triple extraction quality is bounded by the capabilities of $\mathcal{M}_2$. Errors in entity normalization or relation classification by the extractor propagate silently into verification statistics.

5. **Static Ontology Alignment**: Entity names in the KG are the product of a one-time normalization pass. Emerging drug names, novel disease classifications (e.g., post-COVID conditions), or ontology updates (MONDO, DrugBank releases) are not automatically reflected.

6. **Relation Granularity**: The four-class relation schema (`indication`, `contraindication`, `off-label use`, `not_indication`) collapses clinical nuances such as dosage sensitivity, population specificity, and combination-therapy context into binary categories.

7. **API Rate Limits and Latency**: The mandatory 4-second inter-call delay bounds throughput to approximately 120–180 questions per hour, making large-scale (>1000-question) experiments time-consuming.

8. **No Question Type Tracking in Output**: The generated question type label (Type 1–5) is not saved in the output CSV by default, preventing per-category hallucination breakdown without additional instrumentation.

---

## 10. Future Work

1. **Multi-hop Reasoning Verification**: Extend the verification module to support graph-traversal-based validation of multi-step clinical claims using path queries over the KG.

2. **Cross-Model Comparative Study**: Systematically benchmark hallucination rates across a broader LLM family (e.g., GPT-4o, Gemini 1.5 Pro, Mistral-Large, BioMedLM) while holding the prompt set and KG constant.

3. **Dynamic KG Updating**: Integrate automated pipelines to pull updated drug approval data from FDA FAERS, DrugBank REST API, and UMLS Metathesaurus to keep the ground-truth KG current.

4. **Confidence-Weighted Verification**: Incorporate the fuzzy match score returned by `process.extractOne` as a continuous confidence weight in the verification outcome, enabling soft rather than hard verification decisions.

5. **LLM Self-Consistency Evaluation**: Sample each question $k$ times and measure consistency of the extracted triples across repeated queries, providing a complementary measure of uncertainty alongside factual accuracy.

6. **Question Type Tracking**: Persist the question type label in the results CSV and break down hallucination rates per category to identify which prompt patterns (adversarial, open-ended, confirmation) elicit the most hallucinations.

7. **Relation Expansion**: Incorporate additional biomedical relation types beyond drug–disease pairs, including gene–disease, drug–drug interaction, and drug–protein target relationships, for a broader hallucination taxonomy.

8. **Human-in-the-Loop Validation**: Establish a clinical expert annotation layer to adjudicate `Unverifiable` claims, providing ground-truth labels for KG-coverage gaps and enabling precision/recall calculation independent of KG completeness.

---

## 11. Project Structure

```
hallucination project/
│
├── code/
│   ├── config.py               # API client, model IDs
│   ├── main.py                 # Orchestration: question gen, LLM querying, verification, CSV export
│   ├── utils.py                # ask_llm(), extract_triples() with few-shot prompting
│   ├── disease_drug_graph.py   # PrimeKG → graph.gexf builder
│   ├── hetionet_graph.py       # Hetionet v1.0 → GEXF builder
│   └── graph_test.py           # Ad-hoc graph inspection utilities
│
├── hetionet/
│   └── hetionet-v1.0.json      # Hetionet raw data (746 MB, not in version control)
│
├── results/
│   ├── experiment_results_llama-3.3-70b-versatile.csv
│   └── experiment_results_qwen3-32b.csv
│
├── merge.py                    # Multi-source KG merge with fuzzy deduplication
├── wikidata.py                 # Wikidata SPARQL → drug-disease CSV
├── umls_graph.py               # SciSpacy UMLS CUI annotation
├── wikidata_drugs_diseases.csv # Cached Wikidata edges
├── graph.gexf                  # PrimeKG-derived base graph
├── merged_graph.gexf           # Unified multi-source KG (primary input to main.py)
├── graph_with_umls.gexf        # UMLS-enriched variant
├── EXPERIMENT_GUIDE.md         # Step-by-step experiment reproduction guide
└── README.md                   # This document
```

---

## 12. References

1. Yasunaga, M. et al. (2022). *Deep Bidirectional Language-Knowledge Graph Pretraining.* NeurIPS 2022. — Knowledge graph integration with language models.

2. Himmelstein, D. S. et al. (2017). *Systematic integration of biomedical knowledge prioritizes drugs for repurposing.* eLife 6:e26726. — Hetionet knowledge graph.

3. Chandak, P. et al. (2023). *Building a knowledge graph to enable precision medicine.* Scientific Data, 10(67). — PrimeKG precision medicine knowledge graph.

4. Bodenreider, O. (2004). *The Unified Medical Language System (UMLS): Integrating biomedical terminology.* Nucleic Acids Research, 32(Database issue), D267–D270. — UMLS ontology.

5. Neumann, M. et al. (2019). *ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing.* Workshop on Biomedical NLP (BioNLP), ACL 2019. — SciSpacy for biomedical NER and UMLS linking.

6. Ji, Z. et al. (2023). *Survey of Hallucination in Natural Language Generation.* ACM Computing Surveys, 55(12), 1–38. — Comprehensive survey of LLM hallucination types and detection methods.

7. Singhal, K. et al. (2023). *Large Language Models Encode Clinical Knowledge.* Nature, 620, 172–180. — Evaluation of LLMs on clinical knowledge benchmarks.

8. Vrandečić, D. & Krötzsch, M. (2014). *Wikidata: A Free Collaborative Knowledgebase.* Communications of the ACM, 57(10), 78–85. — Wikidata as biomedical knowledge source.

9. RapidFuzz library: https://github.com/maxbachmann/RapidFuzz — Fuzzy string matching for entity resolution.

10. Groq API Documentation: https://console.groq.com/docs — LLM inference API used for querying and triple extraction.

---

*This project is developed for academic research purposes. All LLM outputs and verification results are intended for systematic evaluation and should not be interpreted as clinical medical advice.*
