import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import time

def get_disease_symptoms_paginated():
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setReturnFormat(JSON)
    
    limit = 10
    offset = 0
    all_data = []
    
    print("Starting paginated extraction from Wikidata...")
    
    while True:
        print(f"Fetching rows {offset} to {offset + limit}...")
        
        # We inject the dynamic LIMIT and OFFSET into the query
        query = f"""
        SELECT ?disease ?diseaseLabel ?symptom ?symptomLabel
        WHERE {{
          ?disease wdt:P31/wdt:P279* wd:Q12136.
          ?disease wdt:P1479 ?symptom.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {limit}
        OFFSET {offset}
        """
        
        sparql.setQuery(query)
        
        try:
            results = sparql.query().convert()
            bindings = results["results"]["bindings"]
            
            # If the chunk comes back empty, we've reached the end of the data
            if not bindings:
                print("No more results found. Extraction complete!")
                break
                
            for result in bindings:
                all_data.append({
                    "disease_id": result["disease"]["value"].split('/')[-1],
                    "disease_name": result.get("diseaseLabel", {}).get("value", "Unknown"),
                    "symptom_id": result["symptom"]["value"].split('/')[-1],
                    "symptom_name": result.get("symptomLabel", {}).get("value", "Unknown")
                })
                
            offset += limit
            
            # Be polite to the server to avoid getting IP banned
            time.sleep(2) 
            
        except Exception as e:
            print(f"Error at offset {offset}: {e}")
            print("Retrying in 10 seconds...")
            time.sleep(10) # Wait a bit longer before retrying on an error
            # Note: We don't increase the offset here so it retries the same chunk

    # Convert to DataFrame and save
    if all_data:
        df = pd.DataFrame(all_data)
        # Drop any potential duplicates caused by overlapping offsets
        df = df.drop_duplicates() 
        df.to_csv("wikidata_disease_symptoms_complete.csv", index=False)
        print(f"Successfully saved {len(df)} unique disease-symptom edges to CSV!")
        return df
    else:
        print("Failed to retrieve any data.")
        return None

# Run the extraction
df_kg = get_disease_symptoms_paginated()