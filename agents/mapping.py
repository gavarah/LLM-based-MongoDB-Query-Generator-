from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

class MappingAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a database schema expert.
            Your task is to identify likely field mappings (relationships) between multiple database collections based on their field names.
            
            Collections and their fields:
            {schema_info}
            
            Identify fields in one collection that likely correspond to fields in another collection (e.g., foreign keys, same data).
            - 'user_id' in 'orders' might map to 'id' in 'users'.
            - 'email' in 'users' might map to 'email_address' in 'contacts'.
            
            Return a JSON list of objects, where each object represents a relationship:
            [
                {{
                    "c1": "collection_name_1",
                    "c2": "collection_name_2",
                    "mapping": {{ "field_in_c1": "field_in_c2" }}
                }},
                ...
            ]
            
            Only include mappings you are confident about.
            JSON Output:
            """
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def suggest_mappings(self, collections_schema):
        # collections_schema is a dict: { "col_name": ["field1", "field2"], ... }
        schema_str = ""
        for col, fields in collections_schema.items():
            schema_str += f"Collection: {col}\nFields: {', '.join(fields)}\n\n"
            
        try:
            response = self.chain.invoke({
                "schema_info": schema_str
            })
            
            cleaned_response = response.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_response)
        except Exception as e:
            print(f"Mapping error: {e}")
            return []
