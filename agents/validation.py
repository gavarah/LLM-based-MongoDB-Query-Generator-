from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

class ValidationAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a data validation expert.
            Analyze the following sample data retrieved from a database query to see if it answers the user's natural language query.
            
            User Query: {query}
            
            Sample Data:
            {data}
            
            Does this data look valid and does it answer the user's query?
            If it is empty, looks like an error message, or does not answer the query, mark it as invalid.
            If it looks like legitimate data records that answer the query, mark it as valid.
            
            Respond in JSON format with two keys:
            - "is_valid": boolean
            - "message": string explanation
            """
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def validate(self, data, nl_query):
        try:
            # Convert data to string for prompt
            data_str = json.dumps(data, default=str)[:2000] # Truncate if too long
            response = self.chain.invoke({"data": data_str, "query": nl_query})
            
            # Clean up response if it contains markdown code blocks
            cleaned_response = response.replace("```json", "").replace("```", "").strip()
            
            # Attempt to find JSON object using regex if simple cleanup fails
            import re
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                cleaned_response = json_match.group(0)
            
            return json.loads(cleaned_response)
        except Exception as e:
            return {"is_valid": False, "message": f"Validation error: {str(e)}"}
