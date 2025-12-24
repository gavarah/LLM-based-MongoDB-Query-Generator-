from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import re

class PlannerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are an expert database query planner.
            Your task is to create a step-by-step execution plan to answer a natural language query.
            
            Context:
            - Database Schema (JSON Format): 
            {schema_info}
            
            - Field Mapping (Left -> Right): {mapping}
            
            User Query: {query}
            
            Custom Instructions:
            {custom_instructions}
            
            Instructions:
            1. **CRITICAL**: Analyze the 'user_hints' object in the schema. If 'priority_collections' are listed, you **MUST** use them as the primary source. **IGNORE** any other collections that might seem relevant based on name similarity.
            2. **STRICT SCHEMA ADHERENCE**: Use **ONLY** the collection names and fields explicitly defined in the "collections" object of the provided JSON schema. Do NOT hallucinate or invent field names. If a field is not in the "collections" list, it does not exist.
            3. **FORBIDDEN FIELD**: Do NOT use the `_id` field in any stage of the pipeline (match, group, project, lookup, etc.). Use other unique identifiers available in the schema.
            4. If the user asks for a field that is not in the schema, try to find the closest matching field from the provided list.
            5. **NO UNREQUESTED FILTERS**: Do NOT add any `$match` stages or filters unless the user explicitly mentions a condition (e.g., "where status is active"). If the user just asks to "list" or "show", retrieve ALL records.
            6. Construct a valid MongoDB aggregation pipeline to retrieve the answer.
            7. Identify the starting collection for the aggregation.
            8. Do NOT add a $limit stage unless the user explicitly asks for a specific number of results (e.g., "top 10"). Return ALL matching results by default.
            9. Follow any 'Custom Instructions' provided above strictly.
            
            Return a JSON object with three keys:
            - "explanation": A brief step-by-step text explanation of the plan (e.g., "1. Filter by status... 2. Group by date...").
            - "collection": The name of the collection to run the aggregation on.
            - "pipeline": The valid JSON array for the aggregation pipeline.
            
            Example Output 1 (Filtering & Grouping):
            {{
              "explanation": "1. Match active users. 2. Count by region.",
              "collection": "users",
              "pipeline": [{{"$match": {{"status": "active"}}}}, {{"$group": {{"_id": "$region", "count": {{"$sum": 1}}}}}}]
            }}
            
            Example Output 2 (Simple Projection/Listing):
            {{
              "explanation": "Retrieve all documents and project only the project_code field.",
              "collection": "projects",
              "pipeline": [{{"$project": {{"project_code": 1, "_id": 0}}}}]
            }}
            
            Do not include markdown formatting or explanations outside the JSON.
            """
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def plan(self, query, mapping, schema_info, custom_instructions="", max_retries=2):
        last_error = None
        last_response = ""
        
        for attempt in range(max_retries):
            try:
                response = self.chain.invoke({
                    "query": query,
                    "mapping": str(mapping), # Pass full mapping list as string
                    "schema_info": schema_info,
                    "custom_instructions": custom_instructions
                })
                last_response = response
                
               # Parse schema to get valid collections for validation
                valid_collections = []
                priority_collections = []
                try:
                    schema_data = json.loads(schema_info)
                    valid_collections = list(schema_data.get("collections", {}).keys())
                    priority_collections = schema_data.get("user_hints", {}).get("priority_collections", [])
                except:
                    pass

                # Ensure we return a dict, not a string
                try:
                    # Strategy 1: Extract all markdown code blocks
                    # Use a more flexible regex to catch blocks with different spacing/languages
                    code_blocks = re.findall(r'```\w*\s*(.*?)```', response, re.DOTALL)
                    
                    parsed_blocks = []
                    for block in code_blocks:
                        try:
                            parsed = json.loads(block.strip())
                            parsed_blocks.append(parsed)
                        except:
                            continue
                    
                    # Priority 1: Find a valid JSON Object (Dict) containing "pipeline"
                    final_plan = None
                    for parsed in parsed_blocks:
                        if isinstance(parsed, dict) and "pipeline" in parsed:
                            final_plan = parsed
                            break
                    
                    # Priority 2: Fallback - Find a JSON Array (List) and extract context from text
                    if not final_plan:
                        for parsed in parsed_blocks:
                            if isinstance(parsed, list):
                                # This is likely the pipeline. Try to extract collection from text.
                                # Regex handles: "Collection:", "**Collection**:", "Starting Collection:", etc.
                                coll_match = re.search(r'(?:Starting )?Collection(?: Name)?[:\s*]*\*+`?(\w+)`?', response, re.IGNORECASE)
                                if not coll_match:
                                     # Stricter regex: Must have a colon
                                     coll_match = re.search(r'Collection[:\s]*[:]+[\s]*`?(\w+)', response, re.IGNORECASE)
                                     
                                collection = coll_match.group(1) if coll_match else "unknown"
                                
                                # Try to extract explanation
                                # Look for "Explanation:" or "Steps:" followed by text, up to the start of the code block or end of string
                                expl_match = re.search(r'(?:Explanation|Steps)[:\s*]+(.*?)(?=```|$)', response, re.IGNORECASE | re.DOTALL)
                                explanation = expl_match.group(1).strip() if expl_match else "See generated pipeline."
                                
                                final_plan = {
                                    "explanation": explanation,
                                    "collection": collection,
                                    "pipeline": parsed
                                }
                                break
                    
                    # Strategy 2: Regex for the specific JSON object structure (outside code blocks)
                    if not final_plan:
                        json_match = re.search(r'(\{[\s\S]*"explanation"[\s\S]*\})', response)
                        if json_match:
                            try:
                                final_plan = json.loads(json_match.group(1))
                            except:
                                pass
                    
                    # Strategy 3: Naive cleanup (fallback)
                    if not final_plan:
                        cleaned_response = response.replace("```json", "").replace("```", "").strip()
                        final_plan = json.loads(cleaned_response)
                    
                    # --- VALIDATION & CORRECTION ---
                    # Ensure the extracted collection is valid. If not, use priority hint.
                    if final_plan:
                        extracted_coll = final_plan.get("collection", "unknown")
                        
                        # If extracted collection is NOT in schema (and we have a schema), override it
                        if valid_collections and extracted_coll not in valid_collections:
                            if priority_collections:
                                final_plan["collection"] = priority_collections[0]
                            elif extracted_coll == "unknown" and valid_collections:
                                final_plan["collection"] = valid_collections[0]
                                
                        # If extracted collection IS valid but we have a priority hint that differs?
                        # The prompt says MUST use priority. Let's enforce it if the extracted one is different
                        # but only if the extracted one is NOT one of the priority ones.
                        if priority_collections and extracted_coll not in priority_collections:
                             final_plan["collection"] = priority_collections[0]
                    
                    return final_plan
                    
                except Exception as e:
                        print(f"Attempt {attempt+1} failed: {e}")
                        last_error = e
                        continue
            
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                last_error = e
                continue
        
        # If we get here, all retries failed. Generate Fallback.
        print(f"Planner failed after {max_retries} attempts. Using fallback.")
        return self._create_fallback_plan(schema_info, str(last_error))

    def _create_fallback_plan(self, schema_info, error_msg):
        """Generates a safe fallback plan if the LLM fails."""
        coll = "unknown"
        try:
            schema_data = json.loads(schema_info)
            # Try hints first
            hints = schema_data.get("user_hints", {}).get("priority_collections", [])
            if hints:
                coll = hints[0]
            else:
                # Pick first collection
                colls = list(schema_data.get("collections", {}).keys())
                coll = colls[0] if colls else "unknown"
        except:
            pass
            
        return {
            "explanation": f"Automatic fallback plan generated due to AI error: {error_msg}. Fetching sample documents.",
            "collection": coll,
            "pipeline": [{"$limit": 10}],
            "error": error_msg # Keep track of the error
        }
