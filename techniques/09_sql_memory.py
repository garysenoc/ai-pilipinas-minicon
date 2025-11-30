"""
Technique 9: SQL-Backed Memory (LCEL Pattern)
==============================================

This technique stores conversation history in a SQL database, providing
persistent storage with query capabilities. Good for audit trails and
analytics. Uses modern LCEL pattern.

Pros:
- Persistent storage
- Queryable (can analyze conversations)
- Good for audit trails
- Standard database features (backups, replication)
- Uses modern LangChain v1.0+ patterns (no deprecation warnings)

Cons:
- Slower than in-memory or Redis
- Requires database setup
- More complex queries

Use Case: Applications that need to query and analyze conversation history,
or need strong persistence guarantees.
"""

from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os
import sys
from typing import Dict

# Add parent directory to path for utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.token_counter import (
    count_tokens, 
    count_messages_tokens,
    print_token_stats,
    print_token_summary
)

load_dotenv()

# Store for SQL chat message histories
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create SQL chat message history for a session."""
    if session_id not in store:
        database_url = os.getenv("DATABASE_URL", "sqlite:///conversation_history.db")
        store[session_id] = SQLChatMessageHistory(
            connection=database_url,
            session_id=session_id
        )
    return store[session_id]

def create_sql_memory_agent():
    """Create an agent with SQL-backed memory using LCEL pattern."""
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create a prompt template with message history placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Have a natural conversation with the user."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Create the chain using LCEL
    chain = prompt | llm
    
    # Wrap with message history (SQL-backed)
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    return chain_with_history

def demonstrate_sql_memory():
    """Demonstrate SQL memory using LCEL pattern."""
    print("=" * 60)
    print("Technique 9: SQL-Backed Memory (LCEL Pattern)")
    print("=" * 60)
    print("Using modern LangChain v1.0+ patterns with RunnableWithMessageHistory")
    print()
    
    chain = create_sql_memory_agent()
    session_id = "demo_session"
    config = {"configurable": {"session_id": session_id}}
    
    # Simulate a conversation
    conversations = [
        "Hi, I'm Henry",
        "I'm a product manager",
        "I manage a team of 5 developers",
        "What's my name?",
        "What's my role?",
        "How many people are in my team?"
    ]
    
    total_input_tokens = 0
    total_output_tokens = 0
    
    for i, user_input in enumerate(conversations, 1):
        print(f"User: {user_input}")
        
        # Count input tokens (user message + history)
        input_tokens = count_tokens(user_input)
        history = get_session_history(session_id)
        if history.messages:
            input_tokens += count_messages_tokens(history.messages)
        total_input_tokens += input_tokens
        
        response = chain.invoke(
            {"input": user_input},
            config=config
        )
        print(f"Agent: {response.content}")
        
        # Count output tokens
        output_tokens = count_tokens(response.content)
        total_output_tokens += output_tokens
        
        # Count current memory tokens
        history = get_session_history(session_id)
        memory_tokens = count_messages_tokens(history.messages) if history.messages else 0
        
        print_token_stats(input_tokens, output_tokens, memory_tokens)
        print()
    
    # Show the stored memory
    print("\n" + "-" * 60)
    print("Stored Memory (in SQL Database):")
    print("-" * 60)
    history = get_session_history(session_id)
    print(f"Session ID: {session_id}")
    print(f"Messages: {len(history.messages)}")
    print(f"Database: {os.getenv('DATABASE_URL', 'sqlite:///conversation_history.db')}")
    print()
    
    # Show total token usage
    final_memory = count_messages_tokens(history.messages) if history.messages else 0
    print_token_summary(
        total_input_tokens, 
        total_output_tokens, 
        final_memory
    )

if __name__ == "__main__":
    demonstrate_sql_memory()

