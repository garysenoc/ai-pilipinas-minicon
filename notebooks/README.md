# Conversational History Techniques - Jupyter Notebooks

This folder contains Jupyter notebooks for each conversational history technique implemented in this project.

## Notebooks

Each notebook provides:

1. **Overview** - Brief explanation of the technique
2. **Pros & Cons** - Advantages and limitations
3. **Use Case** - When to use this technique
4. **Implementation** - Complete code for the technique
5. **Example Usage** - How to run the demonstration

## Available Notebooks

1. **01_basic_buffer_memory.ipynb** - Basic conversation buffer memory (stores all messages)
2. **02_summary_memory.ipynb** - Conversation summary memory (maintains running summary)
3. **03_window_memory.ipynb** - Buffer window memory (sliding window of recent messages)
4. **04_summary_buffer_memory.ipynb** - Hybrid summary + buffer memory
5. **05_entity_memory.ipynb** - Entity memory (extracts and stores entities)
6. **06_knowledge_graph_memory.ipynb** - Knowledge graph memory (stores relationships)
7. **07_vector_store_memory.ipynb** - Vector store memory (semantic search)
8. **08_redis_memory.ipynb** - Redis-backed memory (persistent, distributed)
9. **09_sql_memory.ipynb** - SQL-backed memory (queryable, persistent)
10. **10_custom_memory.ipynb** - Custom memory implementation example
11. **11_lcel_memory.ipynb** - LCEL pattern demonstration

## Running the Notebooks

### Prerequisites

1. Install Jupyter:
```bash
pip install jupyter
```

2. Make sure you have all dependencies installed:
```bash
pip install -r requirements.txt
```

3. Set up your `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

### Running a Notebook

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to the `notebooks/` folder

3. Open any notebook (e.g., `01_basic_buffer_memory.ipynb`)

4. Run the cells in order:
   - First cell: Overview and explanation
   - Second cell: Implementation code
   - Third cell: Example usage instructions
   - Fourth cell: Run the demonstration function

## Notes

- All notebooks use modern LangChain v1.0+ patterns with `RunnableWithMessageHistory`
- Each notebook is self-contained and can be run independently
- The code includes token counting utilities to monitor usage
- Make sure your `.env` file is in the project root directory

## Tips

- Start with `01_basic_buffer_memory.ipynb` to understand the basics
- Each technique builds on concepts from previous ones
- Experiment with different parameters to see how they affect behavior
- Check token usage to understand the cost implications of each technique

