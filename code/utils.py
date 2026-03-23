from config import client, MODEL_ID

def ask_llm(question):
    
    response = client.responses.create(
        input=f"You are a medical assistant. Answer this query concisely: {question}",
        model=MODEL_ID
    )
    return response.output_text




def extract_triples(input_text):
    """
    Takes a medical string and uses Gemini to extract (Subject, Relation, Object) triples.
    """

        # 1. System Message (The strict rules)
    # 1. System Message (The strict rules)
    system_instruction = """
    You are an expert medical data extractor. Your task is to extract information from text to build a structured Disease and Drug knowledge graph that aligns with standard medical ontologies (e.g., MONDO for diseases, DrugBank or RxNorm for drugs).

    Step 1 - Entity detection & Normalization: Identify all diseases, medical conditions, and pharmacological substances/drugs in the raw text.
    CRITICAL: You must normalize all extracted entities to their formal, generic medical and pharmacological terms (e.g., convert "Advil" to "ibuprofen", "high blood pressure" to "hypertension", "Tylenol" to "acetaminophen"). Output ALL entities in strict lowercase. Do not extract irrelevant entities like names of people, locations, or dates.
    Step 2 - Coreference resolution: Find all expressions in the text that refer to the same entity. Ensure entities are not duplicated. Only include the most specific version of the entity.
    Step 3 - Relation extraction: Identify clinical relationships between the diseases and drugs you have identified.
    Step 4 - Relationship Categorization: Pay close attention to clinical context. Determine if a drug is an approved treatment, an unapproved but accepted treatment, or if it is medically prohibited for a specific condition.
   Step 5 - Strict Contraindication Rules: A "contraindication" STRICTLY means a known, documented medical danger (e.g., "Drug X causes severe bleeding in Disease Y"). 
    CRITICAL: If the text simply states that a drug is ineffective, not FDA-approved, not a standard treatment, or debunks an internet rumor/fake drug, DO NOT extract a triplet for it. Simply ignore it.
    CRITICAL: To ensure database compatibility with our graph, you MUST strictly use only the following standardized relationships:
    - "indication" (e.g., disease -> indication -> drug)
    - "contraindication" (e.g., disease -> contraindication -> drug)
    - "off-label use" (e.g., disease -> off-label use -> drug)

    Format: Return the knowledge graph as a list of triples, i.e., ["disease entity", "relation", "drug entity"], in Python code.
    """

    # 2. Human Message Examples (The few-shot templates)
    human_message_examples = """
    Here are some example input and output pairs. Note how colloquial diseases are formalized, brand-name drugs are converted to generics, and different clinical relationships are captured.

    Example 1.
    Input:
    "Patients presenting with severe hypertension are frequently prescribed Norvasc or lisinopril to manage their blood pressure."
    Output:
    <python>
    [
    ["hypertension", "indication", "amlodipine"],
    ["hypertension", "indication", "lisinopril"]
    ]
    </python>

    Example 2.
    Input:
    "While Ozempic is officially FDA-approved for the management of type 2 diabetes, doctors frequently prescribe it for weight loss in patients with obesity."
    Output:
    <python>
    [
    ["type 2 diabetes", "indication", "semaglutide"],
    ["obesity", "off-label use", "semaglutide"]
    ]
    </python>

    Example 3.
    Input:
    "A patient with a history of active peptic ulcer disease should not take Advil or Aleve due to the risk of severe gastrointestinal bleeding."
    Output:
    <python>
    [
    ["peptic ulcer disease", "contraindication", "ibuprofen"],
    ["peptic ulcer disease", "contraindication", "naproxen"]
    ]
    </python>
    """

    # 3. Human Message Instructions (The actual task)
    human_message_instructions = f"""Use the given format to extract information from the following input: {input_text}.
    Skip the preamble and output the result strictly as a Python list of lists within <python> tags.

    Important Tips:

    1. Make sure all  clinical relationships (treatments, risks, and off-label uses) are included in the knowledge graph.

    2. Normalize all disease entities to formal lowercase terms, and all drug entities to their lowercase generic chemical names.

    3. Each triple must only contain exactly three strings! The first string must be the disease, the second the relation, and the third the drug.

    4. Do not split up related information into separate triples if it changes the medical meaning.

    5. Make sure all brackets and quotation marks are matched.

    6. Before adding a triple to the knowledge graph, check that the concatenated triple makes sense as a logical statement (e.g., "hypertension indication lisinopril"). If not, discard it.

    7. HANDLE NEGATIONS AS CONTRAINDICATIONS: If the input text explicitly states that a drug should not be used, is harmful, or is not recommended for a condition, map this using the "contraindication" relation.
    """

    # Combine the examples and instructions cleanly into the final user prompt
    user_prompt = f" \n\n {human_message_examples}\n\n{human_message_instructions}\n"

    # Call the Gemini API
    response = client.responses.create(
      model="openai/gpt-oss-120b",
      input=[
          {"role": "system", "content": system_instruction},
          {"role": "user", "content": user_prompt}
      ],
      temperature=0.0
  )
    return response.output_text
