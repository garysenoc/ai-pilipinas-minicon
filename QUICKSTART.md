# Quick Start Guide

Get started with conversational history techniques in 5 minutes!

**Note:** This project is compatible with LangChain v1.0+. For new projects, we recommend using Technique 11 (LCEL) which follows the modern LangChain patterns.

## Step 1: Setup

```bash
# Activate virtual environment
source conversational/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure API Key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Step 3: Run Your First Example

```bash
# Run a simple example
python techniques/01_basic_buffer_memory.py
```

## Step 4: Explore All Techniques

```bash
# Run all examples interactively
python run_all_examples.py
```

## Common Use Cases

### I'm starting a new LangChain v1.0+ project
→ Use **Technique 11: LCEL with Message History** (Recommended)

```python
from techniques import create_lcel_memory_agent
chain = create_lcel_memory_agent()
```

### I need simple memory for short conversations
→ Use **Technique 1: Basic Buffer Memory**

```python
from techniques import create_buffer_memory_agent
chain, memory = create_buffer_memory_agent()
```

### I need memory for long conversations
→ Use **Technique 2: Summary Memory** or **Technique 4: Summary Buffer Memory**

```python
from techniques import create_summary_memory_agent
chain, memory = create_summary_memory_agent()
```

### I need to remember specific facts about users
→ Use **Technique 5: Entity Memory**

```python
from techniques import create_entity_memory_agent
chain, memory = create_entity_memory_agent()
```

### I need persistent memory across sessions
→ Use **Technique 8: Redis Memory** or **Technique 9: SQL Memory**

```python
from techniques import create_redis_memory_agent
chain, memory = create_redis_memory_agent(session_id="user_123")
```

## Next Steps

- Read the full [README.md](README.md) for detailed explanations
- Explore individual technique files in `techniques/` directory
- Modify examples to fit your use case
- Combine techniques for advanced scenarios

## Troubleshooting

**Error: OPENAI_API_KEY not found**
- Make sure you created a `.env` file with your API key

**Error: Module not found**
- Run `pip install -r requirements.txt` to install dependencies

**Error: Redis connection failed (Technique 8)**
- Install and start Redis, or skip this technique

**Error: Database connection failed (Technique 9)**
- SQLite should work by default, or configure DATABASE_URL in `.env`

## Need Help?

- Check the README.md for detailed documentation
- Review the code comments in each technique file
- Each technique file has inline documentation explaining how it works

