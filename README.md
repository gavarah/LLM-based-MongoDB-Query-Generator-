# LLM based MongoDB Query Generator Streamlit App

This is a Streamlit application that uses Large Language Models (LLMs) to generate, explain, and execute MongoDB aggregation pipelines from natural language queries. It features a fully agentic workflow, including schema auto-discovery, intelligent query planning, and results validation.

## Features

-   **Multi-LLM Support**: Works with OpenAI (GPT-4), Google Gemini, and Local Llama (via Ollama).
-   **Agentic Workflow**:
    -   **Mapping Agent**: Automatically discovers relationships between collections using schema analysis.
    -   **Planner Agent**: Generates valid MongoDB aggregation pipelines based on your schema and hints.
    -   **Validation Agent**: Verifies that the query results actually answer your specific question.
-   **Visual Schema Mapping**: Interactive UI to map fields across collections for complex queries.
-   **Plan Review & Edit**: Review the generated query plan, edit the aggregation pipeline JSON directly, and refine the explanation before execution.
-   **Execution & Validation**: Safe execution of read-only queries with automatic results validation.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

## Workflow

1.  **Configuration**: 
    -   Select your LLM provider (OpenAI, Gemini, or Ollama) and enter API keys.
    -   Connect to your MongoDB instance using a connection string.
    -   (Optional) Provide global instructions for the planner.

2.  **Schema Mapping**: 
    -   Use **Auto-Discover** to let the AI find relationships between your collections.
    -   Manually add or refine mappings between collections to help the planner understand your data structure.

3.  **Mapping Review**: 
    -   Review the active mappings and inspected schemas to ensure accuracy.

4.  **Query Input**: 
    -   Enter your natural language question.
    -   (Optional) Provide hints about relevant collections or fields to guide the planner.

5.  **Plan Review**: 
    -   The **Planner Agent** generates a step-by-step explanation and a MongoDB aggregation pipeline.
    -   Review the explanation and the JSON pipeline. You can edit the pipeline manually if needed.

6.  **Execution & Validation**: 
    -   The app executes the pipeline against your MongoDB database.
    -   The **Validation Agent** checks a sample of the results to ensure they answer your original query.
    -   Download the full results as JSON.

## Notes

-   **MongoDB Connection**: Ensure your MongoDB instance is accessible from the machine running the app.
-   **Local Models**: If using Ollama, ensure the service is running (`ollama serve`) and the selected model (default: `llama3`) is pulled.
