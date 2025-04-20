# RAG with Gemini & Qdrant Examples

This Jupyter Notebook (`RAG_with_Gemini_&_Qdrant.ipynb`) provides practical examples and implementations of various Retrieval-Augmented Generation (RAG) patterns. It utilizes Google's Gemini API for powerful language generation and embeddings, and Qdrant as the vector database for efficient semantic search and retrieval.

The notebook explores different strategies for retrieving external knowledge, augmenting prompts, and generating final responses to enhance the accuracy, relevance, and trustworthiness of Large Language Models (LLMs).

## Concepts Demonstrated

This notebook covers a wide range of RAG techniques, categorized by different aspects of the pipeline:

**1. Based on Retrieval Type:**
    *   **Dense Retrieval:** Using semantic similarity search with Gemini embeddings and Qdrant.
    *   **Sparse Retrieval:** Using traditional keyword matching with BM25.
    *   **Hybrid Retrieval:** Combining Dense and Sparse methods, often with re-ranking using a Cross-Encoder model (`sentence-transformers`).

**2. Based on Augmentation Strategy:**
    *   **Pre-Retrieval:** The standard approach - retrieve context first, then generate.
    *   **Post-Retrieval:** Generate an initial answer, retrieve evidence, then verify/refine the answer.
    *   **Iterative Retrieval:** Multi-step retrieval and generation guided by the LLM for complex queries.

**3. Based on Response Generation:**
    *   **Extractive:** Selecting and outputting direct quotes from retrieved context.
    *   **Abstractive:** Summarizing and synthesizing information from context in the LLM's own words.
    *   **Mixed:** Combining abstractive summarization with extractive quotes and citations.

**4. Advanced & Specialized RAG Types:**
    *   **Agent-Based RAG:** Simulating an agent that plans retrieval using multiple tools (local Qdrant search, simulated web search).
    *   **Multi-Modal RAG:** Demonstrating the concept with simulated image input alongside text retrieval to identify an object (e.g., a bird).
    *   **Memory-Augmented RAG:** Incorporating conversation history into the retrieval and generation process for stateful interactions.
    *   **Structured Data RAG:** Using Gemini to generate SQL queries based on natural language to retrieve data from a structured database (SQLite example).
    *   **Graph-Based RAG:** Using Gemini to generate graph queries (Cypher) based on natural language to retrieve data from a knowledge graph (Mock Neo4j example).

## Technologies Used

*   **Google Gemini API:** For LLM generation (`gemini-1.5-pro`, `gemini-1.0-pro`) and text embeddings (`models/embedding-001`).
*   **Qdrant:** Vector database for storing and searching document embeddings (via `qdrant-client`).
*   **`rank-bm25`:** Library for sparse keyword retrieval (BM25 algorithm).
*   **`sentence-transformers`:** Library used here for its `CrossEncoder` model for re-ranking hybrid retrieval results.
*   **Python 3:** The programming language used.
*   **SQLite3:** Standard Python library used for the Structured Data RAG example.
*   **Pillow (PIL):** Used conceptually in the Multi-Modal RAG example for image handling (though the example uses a text description).
*   **NumPy:** For numerical operations, especially in hybrid retrieval scoring.

## Setup Instructions

### 1. Prerequisites
*   Python 3.8 or higher installed.
*   Access to Google Gemini API (requires an API key).
*   A running Qdrant instance (either locally via Docker or using Qdrant Cloud).

### 2. Get the Notebook
*   Clone this repository or download the `RAG_with_Gemini_&_Qdrant.ipynb` file.

### 3. Install Dependencies
*   Open your terminal or command prompt, navigate to the directory containing the notebook, and run:
    ```bash
    pip install qdrant-client google-generativeai rank-bm25 sentence-transformers numpy Pillow
    ```
    *(Note: The notebook's first cell runs this command as well, but it's good practice to install dependencies beforehand).*

### 4. Configure API Keys and Qdrant Host
*   **IMPORTANT:** Do **NOT** hardcode your API keys directly in the notebook. Use environment variables or a secure configuration method.
*   You need to set the following environment variables before running the notebook, or replace the placeholder values within the notebook cells (less secure):
    *   `GEMINI_API_KEY`: Your API key for Google Gemini. Obtainable from Google AI Studio.
    *   `QDRANT_HOST`: The URL of your Qdrant instance (e.g., `http://localhost:6333` for a local Docker instance, or your Qdrant Cloud URL).
    *   `QDRANT_API_KEY`: Your API key for Qdrant Cloud (if using Qdrant Cloud). Leave as `None` or omit if connecting to an unsecured local instance.

    *Example using environment variables (Linux/macOS):*
    ```bash
    GEMINI_API_KEY="AIza..."
    QDRANT_HOST="https://your-qdrant-cluster-url"
    QDRANT_API_KEY="your-qdrant-api-key"
    ```
    *(Windows uses `set` instead of `export`)*

### 5. Qdrant Instance
*   Ensure your Qdrant instance (local or cloud) is running and accessible from where you are running the notebook. The notebook code uses the `QDRANT_HOST` and `QDRANT_API_KEY` variables to connect.
*   The code attempts to create new Qdrant collections for each example (e.g., "Dense-Retrieval", "Hybrid-Retrieval"). If a collection already exists, it will print a message and continue.

## Running the Notebook

1.  Open the `RAG_with_Gemini_&_Qdrant.ipynb` file using Jupyter Notebook, Jupyter Lab, Google Colab, VS Code, or another compatible environment.
2.  Ensure your environment variables (API Keys, Qdrant Host) are set correctly *before* starting the Jupyter kernel.
3.  Run the cells sequentially from top to bottom. Each major section demonstrates a different RAG technique.
4.  Observe the output of each cell, which typically shows:
    *   Confirmation of Qdrant collection setup.
    *   Confirmation of document indexing.
    *   The results of retrieval steps (Dense, Sparse, Hybrid scores).
    *   The final generated response from Gemini based on the specific RAG pattern.

## Notebook Structure

*   **Installation and environment variable:** Installs required libraries. *Requires user to set API keys.*
*   **Dense Retrieval RAG:** Demonstrates basic semantic search using Gemini embeddings and Qdrant.
*   **Sparse Retrieval RAG:** Shows keyword-based search using BM25.
*   **Hybrid Retrieval RAG:** Combines Dense (Qdrant) and Sparse (BM25) results, followed by re-ranking with a Cross-Encoder.
*   **Pre-Retrieval RAG:** Implements the standard RAG flow (retrieve then generate).
*   **Post-Retrieval RAG:** Generates an initial answer, retrieves evidence, then uses Gemini again to verify/refine.
*   **Iterative Retrieval RAG:** Simulates a multi-turn process where Gemini guides retrieval through sub-queries before synthesizing a final answer.
*   **Extractive RAG:** Focuses on prompting Gemini to extract exact sentences/snippets from the retrieved context.
*   **Abstractive RAG:** Prompts Gemini to summarize or explain the retrieved context in its own words.
*   **Mixed RAG:** Combines abstractive summarization with cited extractive quotes from the context.
*   **Agent-Based RAG:** Simulates an agent using Gemini to plan which tools (local Qdrant search, mock web search) to use, executing the plan, and synthesizing a report.
*   **Multi-Modal RAG:** Conceptual example using Gemini's multi-modal capabilities with a *simulated* image description and text context to identify an object. *Requires `Pillow` if using actual images.*
*   **Memory-Augmented RAG:** Implements a chatbot class that uses conversation history to augment retrieval queries and generation prompts.
*   **Structured Data RAG:** Uses Gemini to generate SQL queries from natural language, executes them against an in-memory SQLite database, and synthesizes a response from the results.
*   **Graph-Based RAG:** Uses Gemini to generate Cypher queries from natural language, executes them against a *mock* Neo4j graph database, and synthesizes a response from the relational results.

## Notes and Caveats

*   **API Costs:** Running this notebook will incur costs associated with the Google Gemini API (for embeddings and generation calls). Monitor your usage.
*   **Qdrant Performance:** Qdrant performance depends on your setup (local resources, cloud tier). Indexing large datasets takes time and resources.
*   **Simulations:** The Agent-Based, Multi-Modal (image handling), and Graph-Based sections use simplified simulations or mock components. Real-world implementations would require integrating actual agent frameworks (like LangChain, LlamaIndex), image processing libraries, and graph database drivers (like `neo4j`).
*   **Model Names:** Specific Gemini model names (`models/embedding-001`, `gemini-1.5-pro`) may change. Refer to the official Google Gemini documentation for the latest available models.
*   **Error Handling:** The error handling in the notebook is basic. Production systems require more robust error checking and recovery mechanisms.
*   **Security:** Never commit API keys directly into your code or version control. Use environment variables or secure secret management practices.
*   **Deprecation Warnings:** You might see `DeprecationWarning` for `qdrant.search` – the newer method is `qdrant.query_points`, but `search` was used in some original examples. The Pre-Retrieval example and later ones use `query_points`.
