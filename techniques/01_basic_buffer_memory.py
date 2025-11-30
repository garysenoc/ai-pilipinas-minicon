"""
Technique 1: Basic Conversation Buffer Memory (Modern LCEL Pattern)
====================================================================

This is the simplest form of conversational history using LangChain v1.0+ LCEL.
It stores all messages in a buffer and passes them to the LLM on each call.

Pros:
- Simple and straightforward
- Preserves complete conversation context
- No information loss
- Uses modern LangChain v1.0+ patterns (no deprecation warnings)

Cons:
- Can become expensive with long conversations (more tokens)
- May hit token limits with very long conversations
- No automatic summarization or compression

Use Case: Short to medium-length conversations where you need complete context.
"""

from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
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

# Store for chat message histories
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat message history for a session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def create_buffer_memory_agent():
    """Create an agent with basic buffer memory using modern LCEL pattern."""
    
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
    
    # Wrap with message history (this provides the buffer memory)
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    return chain_with_history

def demonstrate_buffer_memory():
    """Demonstrate basic buffer memory using modern LCEL pattern."""
    print("=" * 60)
    print("Technique 1: Basic Conversation Buffer Memory (LCEL Pattern)")
    print("=" * 60)
    print("Using modern LangChain v1.0+ patterns")
    print()
    
    chain = create_buffer_memory_agent()
    session_id = "demo_session"
    config = {"configurable": {"session_id": session_id}}
    
    # Simulate a conversation
    conversations = [
        "Hi, my name is Alice",
        "What's my name?",
        "I'm a software engineer. What do I do?",
        "What's my name again?"
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
    print("Stored Memory:")
    print("-" * 60)
    history = get_session_history(session_id)
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
    demonstrate_buffer_memory()

