# Medical Knowledge Graph Hallucination Detection

This project constructs a knowledge graph of disease-drug relationships using data from PrimeKG and evaluates Large Language Model (LLM) responses to medical questions for factual accuracy. It extracts relational triples from LLM answers and verifies them against the graph to detect hallucinations or unverifiable claims.

## Project Structure

- `code/`: Core Python scripts
  - `main.py`: Main script for loading the graph, processing questions, and verifying relations
  - `disease_drug_graph.py`: Builds the NetworkX graph from PrimeKG CSV data
  - `config.py`: Configuration for LLM API (using Groq)
  - `utils.py`: Helper functions for LLM queries and triple extraction
- `wikidata.py`: Script for extracting disease-symptom data from Wikidata (optional)
- `graph.gexf`: The constructed knowledge graph in GEXF format
- `results/experiment_results.csv`: Output CSV with verification results

## Features

- **Graph Construction**: Creates a bipartite graph of diseases and drugs with relations like "indication" and "contraindication"
- **LLM Integration**: Uses Groq API to query medical questions and extract structured triples
- **Fuzzy Matching**: Handles entity variations with fuzzy string matching
- **Hallucination Detection**: Verifies extracted relations against the knowledge graph
- **Batch Processing**: Processes multiple questions and logs results

## Installation

1. Clone or download this repository
2. Install Python dependencies:
   ```bash
   pip install networkx pandas openai rapidfuzz sparqlwrapper
   ```
3. Set up environment variables:
   - `GROQ_API_KEY`: Your Groq API key for LLM access
4. Ensure the PrimeKG dataset (`kg.csv`) is available at the specified path in `disease_drug_graph.py`

## Usage

1. **Build the Graph** (if not already done):
   ```bash
   python code/disease_drug_graph.py
   ```
   This creates `graph.gexf` from the PrimeKG CSV.

2. **Run Verification**:
   ```bash
   python code/main.py
   ```
   The script loads the graph, processes predefined questions, and outputs results to `results/experiment_results.csv`.

## Configuration

- Modify `config.py` to change the LLM model or API settings
- Adjust fuzzy matching thresholds in `main.py`
- Add more questions or customize the verification logic in `main.py`

## Dependencies

- NetworkX: Graph construction and manipulation
- Pandas: Data processing
- OpenAI: LLM API client (configured for Groq)
- RapidFuzz: Fuzzy string matching
- SPARQLWrapper: For Wikidata queries (in `wikidata.py`)

## Results

The `experiment_results.csv` contains:
- Original question
- LLM response
- Extracted triples
- Verification status (Verified, Hallucination, Unverifiable, etc.)
- Reason for status

## Contributing

Feel free to extend the graph with additional data sources, improve the triple extraction prompts, or enhance the verification algorithms.

## License

This project is for educational and research purposes. Ensure compliance with data usage licenses for PrimeKG and Wikidata.