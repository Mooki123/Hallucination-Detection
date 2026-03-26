import requests
import pandas as pd

def fetch_wikidata_drug_disease():
    print("Querying Wikidata... (This might take a minute)")
    
    url = 'https://query.wikidata.org/sparql'
    
    # 1. The SPARQL Query
    # We look for both "Drug treats Disease" (P2175) and "Disease treated by Drug" (P2176)
    query = """
    SELECT ?drugLabel ?diseaseLabel ?relation WHERE {
      {
        ?drug wdt:P2175 ?disease.
        BIND("treats" AS ?relation)
      }
      UNION
      {
        ?disease wdt:P2176 ?drug.
        BIND("treated_by" AS ?relation)
      }
      
      # This magical service automatically fetches the English names instead of just Q-codes
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    """
    
    # 2. Wikidata requires a polite "User-Agent" header, or they will block your request
    headers = {
        'User-Agent': 'DrugDiseaseGraphBot/1.0 (https://example.org/bot; your_email@example.com)',
        'Accept': 'application/sparql-results+json'
    }
    
    # 3. Send the request
    response = requests.get(url, params={'query': query}, headers=headers)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
        
    data = response.json()
    
    # 4. Parse the complicated JSON into a clean list of dictionaries
    results = []
    for item in data['results']['bindings']:
        drug_name = item.get('drugLabel', {}).get('value', '').lower()
        disease_name = item.get('diseaseLabel', {}).get('value', '').lower()
        relation = item.get('relation', {}).get('value', '')
        
        # Filter out results where Wikidata didn't have an English label (it returns the Q-code instead)
        if not drug_name.startswith('q') and not disease_name.startswith('q'):
            results.append({
                'x_name': drug_name,
                'x_type': 'drug',
                'y_name': disease_name,
                'y_type': 'disease',
                'relation': relation
            })
            
    # 5. Convert to Pandas DataFrame and drop duplicates
    df = pd.DataFrame(results)
    df = df.drop_duplicates()
    
    return df

# ==========================================
# Execution
# ==========================================

print("Starting extraction...")
wikidata_df = fetch_wikidata_drug_disease()

if wikidata_df is not None:
    output_file = "wikidata_drugs_diseases.csv"
    wikidata_df.to_csv(output_file, index=False)
    
    print(f"\nSuccess! Found {len(wikidata_df)} unique drug-disease relationships.")
    print(f"Data saved to '{output_file}'.")
    
    # Preview the first few rows
    print("\nSample Data:")
    print(wikidata_df.head())