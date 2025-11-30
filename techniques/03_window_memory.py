"""
Technique 3: Conversation Buffer Window Memory (LCEL Pattern)
============================================================

This technique stores only the most recent N messages, creating a sliding
window of conversation history. Older messages are discarded. Uses modern LCEL pattern.

Pros:
- Fixed token usage (predictable costs)
- Simple to implement
- Good for maintaining recent context
- Uses modern LangChain v1.0+ patterns (no deprecation warnings)

Cons:
- Loses older conversation context
- May forget important information from earlier in conversation
- Window size needs to be tuned

Use Case: Conversations where only recent context matters, or when you
need to strictly control token usage.
"""

from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
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
from utils.custom_chat_histories import WindowChatMessageHistory

load_dotenv()

# Store for chat message histories
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str, window_size: int = 3) -> BaseChatMessageHistory:
    """Get or create window chat message history for a session."""
    if session_id not in store:
        store[session_id] = WindowChatMessageHistory(window_size=window_size)
    return store[session_id]

def create_window_memory_agent(window_size=3):
    """Create an agent with window memory using LCEL pattern.
    
    Args:
        window_size: Number of conversation exchanges to keep in memory
    """
    
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
    
    # Wrap with message history (window history)
    def get_history(session_id: str) -> BaseChatMessageHistory:
        return get_session_history(session_id, window_size)
    
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    return chain_with_history

def demonstrate_window_memory():
    """Demonstrate window memory using LCEL pattern."""
    print("=" * 60)
    print("Technique 3: Conversation Buffer Window Memory (LCEL Pattern)")
    print("=" * 60)
    print(f"Window Size: 3 (keeping only last 3 exchanges)")
    print("Using modern LangChain v1.0+ patterns with RunnableWithMessageHistory")
    print()
    
    chain = create_window_memory_agent(window_size=3)
    session_id = "demo_session"
    config = {"configurable": {"session_id": session_id}}
    
    # Simulate a conversation
    conversations = [
        "My name is Charlie",
        "I'm a teacher",
        "I teach mathematics",
        "What's my name?",  # This should fail - name was outside window
        "What do I teach?"  # This should work - teaching is in window
    ]
    
    total_input_tokens = 0
    total_output_tokens = 0
    
    for i, user_input in enumerate(conversations, 1):
        print(f"User: {user_input}")
        
        # Count input tokens (user message + history)
        input_tokens = count_tokens(user_input)
        history = get_session_history(session_id, window_size=3)
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
        
        # Count current memory tokens (windowed - limited size)
        history = get_session_history(session_id, window_size=3)
        memory_tokens = count_messages_tokens(history.messages) if history.messages else 0
        
        print_token_stats(input_tokens, output_tokens, memory_tokens)
        print()
    
    # Show the stored memory (only last k exchanges)
    print("\n" + "-" * 60)
    print("Stored Memory (Last 3 exchanges):")
    print("-" * 60)
    history = get_session_history(session_id, window_size=3)
    for message in history.messages:
        if isinstance(message, HumanMessage):
            print(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"AI: {message.content}")
    print()
    
    # Show total token usage
    final_memory = count_messages_tokens(history.messages) if history.messages else 0
    print_token_summary(
        total_input_tokens, 
        total_output_tokens, 
        final_memory
    )

if __name__ == "__main__":
    demonstrate_window_memory()

