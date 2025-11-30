# Conversational History Techniques with LangChain

This project demonstrates 11 different techniques for implementing conversational history in AI agents using LangChain. Each technique has its own use cases, advantages, and trade-offs.

**Note:** This project is compatible with LangChain v1.0+ and includes both traditional patterns (ConversationChain) and modern LCEL patterns (recommended for v1.0+).

## Table of Contents

1. [Basic Buffer Memory](#1-basic-buffer-memory)
2. [Summary Memory](#2-summary-memory)
3. [Window Memory](#3-window-memory)
4. [Summary Buffer Memory](#4-summary-buffer-memory)
5. [Entity Memory](#5-entity-memory)
6. [Knowledge Graph Memory](#6-knowledge-graph-memory)
7. [Vector Store Memory](#7-vector-store-memory)
8. [Redis-Backed Memory](#8-redis-backed-memory)
9. [SQL-Backed Memory](#9-sql-backed-memory)
10. [Custom Memory](#10-custom-memory)
11. [LCEL with Message History](#11-lcel-with-message-history-langchain-v10)

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key
- (Optional) Redis server for Technique 8
- (Optional) SQL database for Technique 9

### Installation

1. Activate the virtual environment:
```bash
source conversational/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY=your_openai_api_key_here
REDIS_URL=redis://localhost:6379/0  # Optional
DATABASE_URL=sqlite:///conversation_history.db  # Optional
```

## Techniques Overview

### 1. Basic Buffer Memory

**File:** `techniques/01_basic_buffer_memory.py`

The simplest approach - stores all conversation messages in a buffer.

**How it works:**
- All messages are stored in memory
- Complete conversation history is passed to the LLM on each call
- No compression or summarization

**Use Cases:**
- Short to medium-length conversations
- When you need complete context
- Prototyping and development

**Pros:**
- Simple and straightforward
- Preserves complete conversation context
- No information loss

**Cons:**
- Can become expensive with long conversations
- May hit token limits
- No automatic summarization

**Example:**
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
```

---

### 2. Summary Memory

**File:** `techniques/02_summary_memory.py`

Maintains a running summary of the conversation instead of storing all messages.

**How it works:**
- Uses an LLM to summarize conversation history periodically
- Stores summary instead of full messages
- Reduces token usage for long conversations

**Use Cases:**
- Long-running conversations
- When you need to reduce token costs
- Applications with extended user sessions

**Pros:**
- Efficient for long conversations
- Automatically compresses information
- Can handle very long conversation histories

**Cons:**
- Some detail may be lost in summarization
- Requires additional LLM calls
- Summary quality depends on prompt

**Example:**
```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
```

---

### 3. Window Memory

**File:** `techniques/03_window_memory.py`

Stores only the most recent N messages, creating a sliding window.

**How it works:**
- Keeps only the last k conversation exchanges
- Older messages are discarded
- Fixed token usage

**Use Cases:**
- When only recent context matters
- Need to strictly control token usage
- Real-time applications with limited memory

**Pros:**
- Fixed token usage (predictable costs)
- Simple to implement
- Good for maintaining recent context

**Cons:**
- Loses older conversation context
- May forget important information
- Window size needs tuning

**Example:**
```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5)  # Keep last 5 exchanges
```

---

### 4. Summary Buffer Memory

**File:** `techniques/04_summary_buffer_memory.py`

Hybrid approach: summary of older messages + recent messages in full detail.

**How it works:**
- Keeps a summary of older messages
- Maintains recent N messages in full detail
- Balances detail and efficiency

**Use Cases:**
- Long conversations needing both recent detail and older context
- Controlled token usage
- Production applications with extended sessions

**Pros:**
- Balances detail and efficiency
- Maintains recent context in full
- Preserves older context through summarization

**Cons:**
- More complex than individual techniques
- Still requires summarization calls
- Need to tune both window size and summary frequency

**Example:**
```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=200
)
```

---

### 5. Entity Memory

**File:** `techniques/05_entity_memory.py`

Extracts and stores entities (people, places, things) separately.

**How it works:**
- Extracts entities from conversation
- Stores facts about each entity
- Allows querying specific entities

**Use Cases:**
- Personal assistants
- CRM systems
- Applications needing to remember user preferences
- Customer service bots

**Pros:**
- Explicitly tracks important entities
- Good for remembering user details
- Structured memory format
- Can query specific entities

**Cons:**
- Only stores entity-related information
- May miss non-entity context
- Requires entity extraction

**Example:**
```python
from langchain.memory import ConversationEntityMemory

memory = ConversationEntityMemory(llm=llm)
```

---

### 6. Knowledge Graph Memory

**File:** `techniques/06_knowledge_graph_memory.py`

Builds a knowledge graph from the conversation, storing relationships between entities.

**How it works:**
- Extracts entities and relationships
- Builds a graph structure
- Can answer complex queries about relationships

**Use Cases:**
- Complex domains with relationships
- Organizational charts
- Product catalogs
- Technical documentation systems

**Pros:**
- Captures relationships between entities
- Can answer complex relationship queries
- Structured representation
- Good for complex domains

**Cons:**
- More complex to implement
- Requires relationship extraction
- May be overkill for simple conversations
- Requires graph storage

**Example:**
```python
from langchain.memory import ConversationKGMemory
from langchain_community.graphs import NetworkxEntityGraph

graph = NetworkxEntityGraph()
memory = ConversationKGMemory(llm=llm, kg=graph)
```

---

### 7. Vector Store Memory

**File:** `techniques/07_vector_store_memory.py`

Stores conversation history in a vector database for semantic search.

**How it works:**
- Converts messages to embeddings
- Stores in vector database
- Retrieves relevant past conversations via semantic search

**Use Cases:**
- Applications with many users/conversations
- Need to retrieve relevant past context
- Long-term memory systems
- Multi-user applications

**Pros:**
- Semantic search capabilities
- Can retrieve relevant past conversations
- Scales to many conversations
- Good for long-term memory

**Cons:**
- Requires vector database setup
- Retrieval may not be perfect
- More complex than simple memory
- Requires embedding model

**Example:**
```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_texts(texts=[], embedding=embeddings)
retriever = vectorstore.as_retriever()
memory = VectorStoreRetrieverMemory(retriever=retriever)
```

---

### 8. Redis-Backed Memory

**File:** `techniques/08_redis_memory.py`

Stores conversation history in Redis for persistent, distributed storage.

**How it works:**
- Uses Redis as the storage backend
- Messages persist across sessions
- Can be shared across multiple instances

**Use Cases:**
- Production applications
- Distributed systems
- Need persistent memory across sessions
- Multi-instance deployments

**Pros:**
- Persistent storage (survives restarts)
- Distributed (shared across instances)
- Fast access
- Good for production

**Cons:**
- Requires Redis server
- Additional infrastructure
- Network latency (if remote)

**Example:**
```python
from langchain_community.chat_message_histories import RedisChatMessageHistory

message_history = RedisChatMessageHistory(
    url="redis://localhost:6379/0",
    session_id="user_123"
)
memory = ConversationBufferMemory(chat_memory=message_history)
```

---

### 9. SQL-Backed Memory

**File:** `techniques/09_sql_memory.py`

Stores conversation history in a SQL database for queryable persistence.

**How it works:**
- Uses SQL database for storage
- Messages stored in tables
- Can query and analyze conversations

**Use Cases:**
- Need to query conversation history
- Analytics and reporting
- Audit trails
- Applications requiring strong persistence

**Pros:**
- Persistent storage
- Queryable (can analyze conversations)
- Good for audit trails
- Standard database features

**Cons:**
- Slower than in-memory or Redis
- Requires database setup
- More complex queries

**Example:**
```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

message_history = SQLChatMessageHistory(
    connection_string="sqlite:///conversations.db",
    session_id="user_123"
)
memory = ConversationBufferMemory(chat_memory=message_history)
```

---

### 10. Custom Memory

**File:** `techniques/10_custom_memory.py`

Demonstrates creating a custom memory class with specific behavior.

**How it works:**
- Extends `BaseChatMemory`
- Implements custom logic (e.g., time-based filtering)
- Can combine multiple techniques

**Use Cases:**
- Need specific memory behavior
- Domain-specific requirements
- Combining multiple approaches
- Custom optimizations

**Pros:**
- Full control over behavior
- Can implement custom logic
- Can combine techniques
- Domain-specific optimizations

**Cons:**
- More development effort
- Need to maintain custom code
- Must handle edge cases

**Example:**
```python
from langchain.memory import BaseChatMemory

class CustomMemory(BaseChatMemory):
    # Implement custom logic
    pass
```

---

### 11. LCEL with Message History (LangChain v1.0+)

**File:** `techniques/11_lcel_memory.py`

Uses LangChain v1.0's modern LCEL (LangChain Expression Language) pattern with `RunnableWithMessageHistory`. This is the **recommended approach** for new projects using LangChain v1.0+.

**How it works:**
- Uses LCEL pipe operator (`|`) for composition
- `RunnableWithMessageHistory` wraps the chain
- Chat message history stored separately
- More flexible and composable than ConversationChain

**Use Cases:**
- New projects using LangChain v1.0+
- When you want to use modern LangChain patterns
- Integration with LangGraph
- Better type safety and composability

**Pros:**
- Modern LangChain v1.0 pattern (recommended)
- More flexible and composable
- Better integration with LangGraph
- Cleaner API with LCEL
- Better type safety

**Cons:**
- Different from older ConversationChain pattern
- Requires understanding of LCEL
- Slightly more setup

**Example:**
```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
```

**Reference:** [LangChain Chat History Documentation](https://python.langchain.com/docs/concepts/chat_history/)

---

## LangChain Version Compatibility

This project supports **LangChain v1.0+**. The techniques use:
- **Techniques 1-10**: Traditional `ConversationChain` pattern (still supported in v1.0)
- **Technique 11**: Modern LCEL pattern (recommended for v1.0+)

For new projects, we recommend using **Technique 11 (LCEL)** as it's the modern, recommended approach. The other techniques are still valid and useful for understanding different memory patterns.

## Running the Examples

### Run Individual Techniques

```bash
# Activate virtual environment
source conversational/bin/activate

# Run a specific technique
python techniques/01_basic_buffer_memory.py
python techniques/02_summary_memory.py
# ... etc
```

### Run All Techniques

```bash
python run_all_examples.py
```

## Comparison Matrix

| Technique | Token Efficiency | Persistence | Complexity | Best For |
|-----------|-----------------|-------------|------------|----------|
| Buffer Memory | Low | No | Low | Short conversations |
| Summary Memory | High | No | Medium | Long conversations |
| Window Memory | Medium | No | Low | Recent context only |
| Summary Buffer | High | No | Medium | Long conversations with recent detail |
| Entity Memory | Medium | No | Medium | Entity-focused apps |
| KG Memory | Medium | No | High | Relationship-heavy domains |
| Vector Store | High | Yes* | High | Many conversations |
| Redis Memory | Low | Yes | Medium | Distributed systems |
| SQL Memory | Low | Yes | Medium | Queryable history |
| Custom Memory | Varies | Varies | High | Specific requirements |
| LCEL Memory | Low | Yes** | Medium | LangChain v1.0+ projects |

*Vector stores can be persisted to disk
**LCEL memory can use any chat message history backend (in-memory, Redis, SQL, etc.)

## Choosing the Right Technique

1. **New LangChain v1.0+ projects**: Use **LCEL Memory (Technique 11)** - recommended
2. **Short conversations (< 10 exchanges)**: Use Buffer Memory
3. **Long conversations**: Use Summary Memory or Summary Buffer Memory
4. **Need only recent context**: Use Window Memory
5. **Remember specific facts**: Use Entity Memory
6. **Complex relationships**: Use Knowledge Graph Memory
7. **Many conversations**: Use Vector Store Memory
8. **Production/distributed**: Use Redis or SQL Memory
9. **Custom requirements**: Create Custom Memory

## Best Practices

1. **Start simple**: Begin with Buffer Memory, upgrade as needed
2. **Monitor token usage**: Track costs and optimize accordingly
3. **Test thoroughly**: Each technique behaves differently
4. **Consider persistence**: Production apps need persistent storage
5. **Tune parameters**: Window sizes, token limits, etc. need tuning
6. **Handle errors**: Memory operations can fail (network, storage, etc.)

## Additional Resources

- [LangChain Chat History (v1.0)](https://python.langchain.com/docs/concepts/chat_history/) - Official documentation
- [LangChain Memory Documentation](https://python.langchain.com/docs/modules/memory/) - Legacy patterns (still supported)
- [LangChain LCEL Documentation](https://python.langchain.com/docs/expression_language/) - Modern LCEL patterns
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) - For advanced agent orchestration
- [OpenAI API Documentation](https://platform.openai.com/docs/)

## License

This project is for educational purposes. Feel free to use and modify as needed.

# ai-pilipinas-minicon
