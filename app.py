import streamlit as st
import pandas as pd
import json
import sqlite3
import time
import certifi
from pymongo import MongoClient
from utils.llm_factory import get_llm
from agents.planner import PlannerAgent
from agents.planner import PlannerAgent
from agents.validation import ValidationAgent
from agents.mapping import MappingAgent

# --- Page Config ---
st.set_page_config(page_title="Query Generator Agent App", layout="wide")

# --- Session State Initialization ---
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'mongo_client' not in st.session_state:
    st.session_state.mongo_client = None
if 'db_names' not in st.session_state:
    st.session_state.db_names = []
if 'selected_db' not in st.session_state:
    st.session_state.selected_db = None
if 'collections' not in st.session_state:
    st.session_state.collections = []
if 'mapping_data' not in st.session_state:
    st.session_state.mapping_data = {}
if 'nl_query' not in st.session_state:
    st.session_state.nl_query = ""
if 'plan' not in st.session_state:
    st.session_state.plan = ""
if 'query_results' not in st.session_state:
    st.session_state.query_results = None
if 'llm_config' not in st.session_state:
    st.session_state.llm_config = {}
if 'all_mappings' not in st.session_state:
    st.session_state.all_mappings = []
if 'last_selected_c1' not in st.session_state:
    st.session_state.last_selected_c1 = None
if 'last_selected_c2' not in st.session_state:
    st.session_state.last_selected_c2 = None

# --- Helper Functions ---
def init_db():
    conn = sqlite3.connect('app_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS mappings
                 (id INTEGER PRIMARY KEY, name TEXT, content TEXT)''')
    conn.commit()
    conn.close()

def save_mapping_to_db(name, mapping):
    conn = sqlite3.connect('app_data.db')
    c = conn.cursor()
    c.execute("INSERT INTO mappings (name, content) VALUES (?, ?)", (name, json.dumps(mapping)))
    conn.commit()
    conn.close()

# Initialize SQLite
init_db()

# --- UI Layout ---
st.title("Query Generator Agent Workflow")

col1, col2 = st.columns([1, 3])

# --- Left Panel (Steps) ---
with col1:
    st.header("Steps")
    
    steps = {
        1: "Configuration",
        2: "Schema Mapping",
        3: "Mapping Review",
        4: "Query Input",
        5: "Plan Review",
        6: "Execution & Validation"
    }
    
    for step_id, step_name in steps.items():
        if step_id == st.session_state.current_step:
            st.markdown(f"**-> {step_id}. {step_name}**")
        elif step_id < st.session_state.current_step:
            st.markdown(f"✓ {step_id}. {step_name}")
        else:
            st.markdown(f"{step_id}. {step_name}")

# --- Right Panel (Inputs & Actions) ---
with col2:
    st.header(steps[st.session_state.current_step])
    
    # STEP 1: Configuration
    if st.session_state.current_step == 1:
        st.subheader("LLM Configuration")
        
        llm_provider = st.selectbox("Select LLM Provider", ["OpenAI", "Gemini", "Local Llama (Ollama)"])
        
        api_key = ""
        model_name = ""
        
        if llm_provider == "OpenAI":
            api_key = st.text_input("OpenAI API Key", type="password")
            model_name = st.text_input("Model Name", value="gpt-4o")
        elif llm_provider == "Gemini":
            api_key = st.text_input("Google API Key", type="password")
            model_name = st.text_input("Model Name", value="gemini-2.5-flash")
        elif llm_provider == "Local Llama (Ollama)":
            model_name = st.text_input("Model Name", value="llama3")
            st.info("Ensure Ollama is running locally.")

        st.divider()
        
        st.subheader("MongoDB Connection")
        conn_string = st.text_input("Enter Connection String", value="mongodb://localhost:27017/")
        
        st.subheader("Global Planner Instructions (Optional)")
        st.info("Provide specific rules or business logic for the AI Planner (e.g., 'Always exclude deleted records').")
        custom_instructions = st.text_area("Custom Instructions:", value=st.session_state.get("custom_instructions", ""), height=100)
        st.session_state.custom_instructions = custom_instructions
        
        if st.button("Connect & Configure"):
            try:
                # Test LLM Connection (Basic check)
                if llm_provider in ["OpenAI", "Gemini"] and not api_key:
                    st.error(f"API Key required for {llm_provider}")
                else:
                    # Save LLM Config
                    st.session_state.llm_config = {
                        "provider": llm_provider,
                        "api_key": api_key,
                        "model_name": model_name
                    }
                    
                    # Test Mongo Connection
                    client = MongoClient(conn_string, serverSelectionTimeoutMS=2000, tlsCAFile=certifi.where())
                    client.server_info() # Trigger connection check
                    st.session_state.mongo_client = client
                    all_dbs = client.list_database_names()
                    st.session_state.db_names = [db for db in all_dbs if db not in ['admin', 'config', 'local']]
                    
                    st.success("Connected successfully!")
                    time.sleep(0.5)
                    st.session_state.current_step = 2
                    st.rerun()
            except Exception as e:
                st.error(f"Connection failed: {e}")

    # STEP 2: Schema Mapping
    elif st.session_state.current_step == 2:
        st.subheader("Create Mapping")
        
        if not st.session_state.mongo_client:
            st.error("No DB Connection. Go back to Step 1.")
            if st.button("Back"):
                st.session_state.current_step = 1
                st.rerun()
        else:
            db_name = st.selectbox("Select Database", st.session_state.db_names)
            st.session_state.selected_db = db_name
            db = st.session_state.mongo_client[db_name]
            db = st.session_state.mongo_client[db_name]
            all_collections = db.list_collection_names()
            
            # --- Global Auto-Discovery ---
            st.write("### Global Auto-Discovery")
            st.info("Click below to analyze ALL collections in the database and discover relationships automatically.")
            
            if st.button("Auto-Discover Relationships (All Collections)"):
                try:
                    llm = get_llm(**st.session_state.llm_config)
                    mapper = MappingAgent(llm)
                    
                    # Fetch schema for all collections (sample first doc)
                    full_schema = {}
                    with st.spinner("Fetching schemas..."):
                        for col in all_collections:
                            doc = db[col].find_one()
                            if doc:
                                full_schema[col] = list(doc.keys())
                    
                    with st.spinner("Analyzing global relationships..."):
                        suggestions = mapper.suggest_mappings(full_schema)
                    
                    # --- Deterministic Exact Match Fallback ---
                    # The AI might miss simple exact matches. Let's find them programmatically.
                    exact_matches = []
                    cols_list = list(full_schema.keys())
                    for i in range(len(cols_list)):
                        for j in range(i + 1, len(cols_list)):
                            c1 = cols_list[i]
                            c2 = cols_list[j]
                            f1 = set(full_schema[c1])
                            f2 = set(full_schema[c2])
                            
                            # Find common fields, excluding standard Mongo '_id'
                            common = f1.intersection(f2)
                            if '_id' in common:
                                common.remove('_id')
                            
                            if common:
                                mapping = {f: f for f in common}
                                exact_matches.append({
                                    "c1": c1,
                                    "c2": c2,
                                    "mapping": mapping
                                })
                    
                    # Merge AI suggestions and Exact Matches
                    # We prefer AI suggestions if they exist for a pair, or we can merge the mappings
                    # Let's simply add exact matches if no mapping exists for that pair yet, 
                    # or merge fields if the pair exists.
                    
                    combined_mappings = {} # Key: (c1, c2) -> mapping_dict
                    
                    # Process AI suggestions first
                    for s in suggestions:
                        c1, c2 = s.get("c1"), s.get("c2")
                        m = s.get("mapping", {})
                        if c1 and c2 and m:
                            # Normalize key order
                            k = tuple(sorted((c1, c2)))
                            if k not in combined_mappings:
                                combined_mappings[k] = {}
                            combined_mappings[k].update(m)

                    # Process Exact Matches
                    for em in exact_matches:
                        c1, c2 = em["c1"], em["c2"]
                        m = em["mapping"]
                        k = tuple(sorted((c1, c2)))
                        if k not in combined_mappings:
                            combined_mappings[k] = {}
                        # Add exact matches if not already present
                        for k_field, v_field in m.items():
                            # We need to be careful about direction since we sorted the key
                            # But for exact match f->f, it doesn't matter direction
                            if k_field not in combined_mappings[k] and v_field not in combined_mappings[k].values():
                                combined_mappings[k][k_field] = v_field

                    # Convert back to list for session state
                    count = 0
                    for (c1, c2), mapping_dict in combined_mappings.items():
                        # Verify collections exist in schema (sanity check)
                        if c1 in full_schema and c2 in full_schema and mapping_dict:
                            # Ensure c1 is the one with keys in mapping_dict (mostly true by construction but let's verify)
                            # For exact matches it's fine. For AI, we trusted its output.
                            # To be safe, we just save it.
                            
                            new_mapping = {
                                "c1": c1,
                                "c2": c2,
                                "fields1": full_schema[c1],
                                "fields2": full_schema[c2],
                                "mapping": mapping_dict
                            }
                            st.session_state.all_mappings.append(new_mapping)
                            count += 1
                            
                    st.success(f"Discovery complete! Added {count} relationships (including exact field matches).")
                    
                except Exception as e:
                    st.error(f"Error during global discovery: {e}")
            
            st.divider()
            
            st.write("### Manual Mapping")
            c1, c2 = st.columns(2)
            with c1:
                coll1_name = st.selectbox("Left Collection", all_collections, key="c1")
            with c2:
                coll2_name = st.selectbox("Right Collection", all_collections, key="c2")
            
            # Fetch sample fields (naively taking first doc keys)
            fields1 = []
            fields2 = []
            
            if coll1_name:
                doc = db[coll1_name].find_one()
                if doc: fields1 = list(doc.keys())
            
            if coll2_name:
                doc = db[coll2_name].find_one()
                if doc: fields2 = list(doc.keys())
                
            st.write("#### Map Fields")
            
            # Simple mapping UI
            mapping = {}
            
            # Determine default selections
            # Start with exact matches
            default_fields = [f for f in fields1 if f in fields2]
            
            # Allow user to select which fields to map
            fields_to_map = st.multiselect("Select fields from Left Collection to map:", fields1, default=default_fields)
            
            for f1 in fields_to_map:
                # Try to find a matching name in fields2 for default
                default_idx = 0
                if f1 in fields2:
                    default_idx = fields2.index(f1) + 1 # +1 because of "None" option
                
                options = ["None"] + fields2
                mapped_f2 = st.selectbox(f"Map '{f1}' to:", options, index=default_idx, key=f"map_{f1}")
                if mapped_f2 != "None":
                    mapping[f1] = mapped_f2
            
            if st.button("Add Relationship"):
                if not mapping:
                    st.warning("Please define at least one field mapping.")
                else:
                    # Add to list of mappings
                    new_mapping = {
                        "c1": coll1_name,
                        "c2": coll2_name,
                        "fields1": fields1,
                        "fields2": fields2,
                        "mapping": mapping
                    }
                    # Check if exists and update, or append
                    # Simple append for now, user can delete in review
                    st.session_state.all_mappings.append(new_mapping)
                    st.success(f"Added relationship between {coll1_name} and {coll2_name}")
            
            st.divider()
            st.write("### Defined Mappings")
            if not st.session_state.all_mappings:
                st.info("No mappings defined yet.")
            else:
                for i, m in enumerate(st.session_state.all_mappings):
                    st.write(f"**{i+1}. {m['c1']} <-> {m['c2']}** ({len(m['mapping'])} fields)")
            
            if st.button("Proceed to Review"):
                if not st.session_state.all_mappings:
                    st.warning("Please add at least one relationship.")
                else:
                    st.session_state.current_step = 3
                    st.rerun()

    # STEP 3: Mapping Review
    elif st.session_state.current_step == 3:
        st.subheader("Review Mappings & Schema")
        
        if not st.session_state.all_mappings:
            st.warning("No mappings defined. Go back to Step 2.")
            if st.button("Back to Mapping"):
                st.session_state.current_step = 2
                st.rerun()
        else:
            st.write("### Active Mappings")
            
            # Allow deleting mappings
            mappings_to_keep = []
            for i, m in enumerate(st.session_state.all_mappings):
                c1, c2 = m['c1'], m['c2']
                with st.expander(f"{c1} <-> {c2}", expanded=True):
                    st.write(f"**Mapped Fields:**")
                    st.json(m['mapping'])
                    if st.checkbox(f"Keep this mapping ({c1}-{c2})", value=True, key=f"keep_{i}"):
                        mappings_to_keep.append(m)
            
            st.session_state.all_mappings = mappings_to_keep

            st.divider()
            st.write("### Schema Information (All Mapped Collections)")
            
            # Aggregate unique collections
            unique_collections = {}
            for m in st.session_state.all_mappings:
                unique_collections[m['c1']] = m['fields1']
                unique_collections[m['c2']] = m['fields2']
            
            cols = st.columns(len(unique_collections))
            for idx, (c_name, c_fields) in enumerate(unique_collections.items()):
                with cols[idx % len(cols)]:
                    st.markdown(f"**{c_name}**")
                    st.dataframe(c_fields, use_container_width=True, hide_index=True, column_config={0: "Field"})
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Confirm & Proceed to Query"):
                    st.session_state.current_step = 4
                    st.rerun()
            with c2:
                if st.button("Back to Add More Mappings"):
                    st.session_state.current_step = 2
                    st.rerun()

    # STEP 4: Query Input
    elif st.session_state.current_step == 4:
        st.subheader("Natural Language Query")
        query = st.text_area("Enter your query:", height=150)
        
        # Optional Hints
        with st.expander("Optional: Provide Query Hints (Collections & Fields)"):
            # Gather all collections and fields from all mappings
            all_mapped_collections = set()
            all_mapped_fields = []
            
            for m in st.session_state.all_mappings:
                all_mapped_collections.add(m['c1'])
                all_mapped_collections.add(m['c2'])
                for f in m['fields1']:
                    all_mapped_fields.append(f"{m['c1']}.{f}")
                for f in m['fields2']:
                    all_mapped_fields.append(f"{m['c2']}.{f}")
            
            # Remove duplicates from fields
            all_mapped_fields = list(set(all_mapped_fields))
            
            # Get ALL collections in the DB for the hint
            all_db_collections = []
            if st.session_state.mongo_client and st.session_state.selected_db:
                db = st.session_state.mongo_client[st.session_state.selected_db]
                all_db_collections = db.list_collection_names()
            
            # Filter defaults to ensure they exist in the available options
            # valid_defaults = [c for c in all_mapped_collections if c in all_db_collections]
            # User requested to start with empty selection for better UX
            hint_collections = st.multiselect("Relevant Collections:", all_db_collections, default=[])
            hint_fields = st.multiselect("Relevant Fields:", all_mapped_fields)
        
        if st.button("Generate Plan"):
            if not query:
                st.warning("Please enter a query.")
            else:
                st.session_state.nl_query = query
                
                try:
                    # Initialize Planner Agent
                    llm = get_llm(**st.session_state.llm_config)
                    planner = PlannerAgent(llm)
                    
                    # Construct Schema Info from All Mappings
                    # Construct Schema Info as Structured JSON for better LLM comprehension
                    schema_data = {
                        "database": st.session_state.selected_db,
                        "user_hints": {
                            "priority_collections": hint_collections,
                            "priority_fields": hint_fields
                        },
                        "collections": {},
                        "relationships": []
                    }
                    
                    # 1. Fetch and List Collections
                    db = st.session_state.mongo_client[st.session_state.selected_db]
                    all_cols = db.list_collection_names()
                    
                    target_cols = []
                    if hint_collections:
                        # STRICT MODE: Only use hinted collections
                        target_cols = [c for c in all_cols if c in hint_collections]
                    else:
                        # Use all collections
                        target_cols = [c for c in all_cols if c not in ['admin', 'config', 'local']]
                        target_cols.sort()
                    
                    with st.spinner("Fetching database schema..."):
                        for c_name in target_cols:
                            if c_name in ['admin', 'config', 'local']: continue
                            
                            try:
                                doc = db[c_name].find_one()
                                if doc:
                                    # Exclude _id from the field list
                                    fields = [k for k in doc.keys() if k != '_id']
                                else:
                                    fields = []
                                schema_data["collections"][c_name] = fields
                            except Exception as e:
                                schema_data["collections"][c_name] = [f"Error: {str(e)}"]
                    
                    # 2. List Relationships
                    if st.session_state.all_mappings:
                        for m in st.session_state.all_mappings:
                            # Filter mappings if hints are present
                            if hint_collections:
                                # Check if either side of the mapping is in the hinted collections
                                if m['c1'] not in hint_collections and m['c2'] not in hint_collections:
                                    continue
                            
                            # Extract just the mapped fields for clarity
                            mapped_pairs = [f"{k}={v}" for k, v in m['mapping'].items()]
                            schema_data["relationships"].append(f"{m['c1']} <-> {m['c2']} ON ({', '.join(mapped_pairs)})")
                    
                    # Convert to JSON string
                    schema_info = json.dumps(schema_data, indent=2)
                        
                    # Debug: View Generated Schema Context
                    with st.expander("Debug: View Context Sent to Planner", expanded=False):
                        st.code(schema_info, language="json")

                    with st.spinner("Generating plan..."):
                        custom_instr = st.session_state.get("custom_instructions", "")
                        plan = planner.plan(query, st.session_state.all_mappings, schema_info, custom_instr)
                    
                    st.session_state.plan = plan
                    st.session_state.current_step = 5
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating plan: {e}")

    # STEP 5: Plan Review
    elif st.session_state.current_step == 5:
        st.subheader("Review & Edit Query Plan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original User Query:**")
            st.info(st.session_state.nl_query)
            
        with col2:
            st.markdown("**Execution Plan & Query:**")
            
            # Parse the plan object
            plan_data = st.session_state.plan
            
            # If it's already a dict (from planner), use it directly
            if isinstance(plan_data, dict):
                pass
            # If it's a string (from previous edits or LLM output parser), parse it
            elif isinstance(plan_data, str):
                try:
                    # Clean up potential markdown code blocks
                    clean_str = plan_data.replace("```json", "").replace("```", "").strip()
                    plan_data = json.loads(clean_str)
                except:
                    plan_data = {"collection": "unknown", "pipeline": [], "explanation": ""}
            else:
                 plan_data = {"collection": "unknown", "pipeline": [], "explanation": ""}
            
            # Check for errors from planner
            if "error" in plan_data:
                st.error(f"Plan Generation Error: {plan_data['error']}")
                with st.expander("Raw Response"):
                    st.code(plan_data.get("raw_response", ""), language="text")
            
            # Extract components
            
            # Extract components
            explanation = plan_data.get("explanation", "")
            # Prioritize the collection from the plan, default to unknown only if missing
            coll_name = plan_data.get("collection", "unknown")
            
            # Robust pipeline extraction
            raw_pipeline = plan_data.get("pipeline", [])
            if isinstance(raw_pipeline, list):
                pipeline = raw_pipeline
            elif isinstance(raw_pipeline, str):
                try:
                    pipeline = json.loads(raw_pipeline)
                except:
                    pipeline = []
            else:
                pipeline = []
            
            # Debug: Always show plan data for troubleshooting
            with st.expander("Debug: Plan Data Keys", expanded=False):
                st.write("Plan Data Keys:", plan_data.keys())
                st.write("Full Plan Data:", plan_data)
                st.write("Raw Pipeline Type:", type(raw_pipeline))
            
            # Display Plan Explanation
            new_explanation = st.text_area("Plan Explanation (Steps):", value=explanation, height=150)
            
            # Display Collection Name
            new_coll = st.text_input("Target Collection:", value=coll_name)
            
            # Display Pipeline (Editable)
            pipeline_str = json.dumps(pipeline, indent=2)
            new_pipeline_str = st.text_area("Aggregation Pipeline (JSON Only)", value=pipeline_str, height=300)
            
            # Update session state with new values
            try:
                new_pipeline = json.loads(new_pipeline_str)
                st.session_state.plan = json.dumps({
                    "explanation": new_explanation,
                    "collection": new_coll, 
                    "pipeline": new_pipeline
                })
            except json.JSONDecodeError:
                st.error("Invalid JSON in pipeline!")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Confirm & Execute"):
                st.session_state.current_step = 6
                st.rerun()
        with c2:
            if st.button("Edit Mapping / Query"):
                st.session_state.current_step = 4 # Go back to Query
                st.rerun()

    # STEP 6: Execution & Validation
    elif st.session_state.current_step == 6:
        st.subheader("Execution & Validation")
        
        status_container = st.empty()
        status_container.write("Executing query...")
        
        try:
            # Parse the plan (Pipeline)
            plan_str = st.session_state.plan.strip()
            if plan_str.startswith("```"):
                plan_str = plan_str.replace("```json", "").replace("```", "").strip()
            
            pipeline = json.loads(plan_str)
            
            # Get DB and Collection
            # Get DB and Collection
            if not st.session_state.selected_db:
                st.error("Missing DB information. Please restart configuration.")
            else:
                db_name = st.session_state.selected_db
                # For execution, we might need to know the 'primary' collection or the pipeline handles it
                # The pipeline usually starts with db.collection.aggregate
                # We need to extract the starting collection from the plan or just try to run it
                
                db = st.session_state.mongo_client[db_name]
                
                # Attempt to infer starting collection from pipeline or user hints
                # For now, let's assume the planner provides a pipeline that starts with a specific collection
                # BUT pymongo aggregate is run ON a collection object: db.collection.aggregate(pipeline)
                # So we need to know WHICH collection to run it on.
                
                # Hack: Ask planner to include "primary_collection" in output? 
                # Or just try to guess from the first $match or $lookup?
                # Better: Let the user confirm the primary collection in Step 5?
                
                # Let's try to find the collection from the hints or the first mapped collection
                if st.session_state.all_mappings:
                    coll_name = st.session_state.all_mappings[0]['c1'] # Default to first
                else:
                    coll_name = "unknown"
                
                # TODO: Ideally the plan should specify the collection.
                # For now, let's assume the pipeline is valid for the first collection in the mapping list
                # OR we can parse the plan to see if it implies a collection.
                
                # Let's check if the plan is a dict with "collection" and "pipeline"
                if isinstance(pipeline, dict) and "collection" in pipeline:
                    coll_name = pipeline["collection"]
                    pipeline = pipeline["pipeline"]
                
                st.write(f"Running aggregation on collection: **{coll_name}**")
                
                # Check for ANY $limit stage in the pipeline
                limit_indices = []
                limit_val = 0
                if isinstance(pipeline, list):
                    for i, stage in enumerate(pipeline):
                        if "$limit" in stage:
                            limit_indices.append(i)
                            limit_val = stage["$limit"] # Just take the last one found for display
                
                has_limit = len(limit_indices) > 0
                
                # Option to ignore limit
                ignore_limit = False
                if has_limit:
                    st.warning(f"The generated query contains a $limit stage (limiting to {limit_val} records).")
                    ignore_limit = st.checkbox("Fetch all results (Remove $limit stages)", value=True)
                
                if ignore_limit and has_limit:
                    # Remove all limit stages
                    pipeline = [stage for i, stage in enumerate(pipeline) if i not in limit_indices]
                    st.info("Limit stages removed. Fetching all results...")
                
                st.write(f"Running aggregation on collection: **{coll_name}**")
                
                # Execute Query
                start_time = time.time()
                results_cursor = db[coll_name].aggregate(pipeline)
                results = list(results_cursor)
                exec_time = time.time() - start_time
                
                status_container.write(f"Query executed in {exec_time:.2f}s. Retrieved {len(results)} records.")
                
                # Take a sample for validation
                result_sample = results[:5] if results else []
                
                if not result_sample:
                    st.warning("Query returned no results.")
                    if st.button("Back to Query"):
                        st.session_state.current_step = 4
                        st.rerun()
                else:
                    # Initialize Validation Agent
                    llm = get_llm(**st.session_state.llm_config)
                    validator = ValidationAgent(llm)
                    
                    st.markdown("**Validating Sample Data:**")
                    st.json(result_sample)
                    
                    with st.spinner("Validating results against query..."):
                        validation_res = validator.validate(result_sample, st.session_state.nl_query)
                    
                    is_valid = validation_res.get("is_valid", False)
                    msg = validation_res.get("message", "No message")
                    
                    if is_valid:
                        st.success(f"Validation Successful: {msg}")
                        st.session_state.query_results = results
                        
                        st.write(f"### Results ({len(results)} records)")
                        
                        # Download Option
                        results_json = json.dumps(results, default=str, indent=2)
                        st.download_button(
                            label="Download Results (JSON)",
                            data=results_json,
                            file_name="query_results.json",
                            mime="application/json"
                        )
                        
                        # Display Data (Scrollable Dataframe)
                        # Convert to simple types for display to avoid ObjectId errors in dataframe
                        display_data = json.loads(results_json)
                        st.dataframe(display_data, use_container_width=True)
                        
                        st.divider()
                        st.write("### Next Steps")
                        
                        col_nav1, col_nav2, col_nav3 = st.columns(3)
                        with col_nav1:
                            if st.button("Modify Mappings (Step 2)"):
                                st.session_state.current_step = 2
                                st.rerun()
                        with col_nav2:
                            if st.button("Review Schema (Step 3)"):
                                st.session_state.current_step = 3
                                st.rerun()
                        with col_nav3:
                            if st.button("New Query (Step 4)"):
                                st.session_state.current_step = 4
                                st.rerun()
                    else:
                        st.error(f"Validation Failed: {msg}")
                        if st.button("Back to Query"):
                            st.session_state.current_step = 4
                            st.rerun()
                    
        except Exception as e:
            st.error(f"Error during execution/validation: {e}")
            if st.button("Back to Plan"):
                st.session_state.current_step = 5
                st.rerun()
