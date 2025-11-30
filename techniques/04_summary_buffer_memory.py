"""
Technique 4: Conversation Summary Buffer Memory (LCEL Pattern)
=============================================================

This is a hybrid approach that combines summary and window memory. It keeps
a summary of older messages and the most recent N messages in full detail.
Uses modern LCEL pattern.

Pros:
- Balances detail and efficiency
- Maintains recent context in full
- Preserves older context through summarization
- Best of both worlds
- Uses modern LangChain v1.0+ patterns (no deprecation warnings)

Cons:
- More complex than individual techniques
- Still requires summarization calls
- Need to tune both window size and summary frequency

Use Case: Long conversations where you need both recent detail and older
context, with controlled token usage.
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
from utils.custom_chat_histories import SummaryBufferChatMessageHistory

load_dotenv()

# Store for chat message histories
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str, max_token_limit: int = 200) -> BaseChatMessageHistory:
    """Get or create summary buffer chat message history for a session."""
    if session_id not in store:
        summary_llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        store[session_id] = SummaryBufferChatMessageHistory(
            summary_llm=summary_llm,
            max_token_limit=max_token_limit
        )
    return store[session_id]

def create_summary_buffer_memory_agent(max_token_limit=200):
    """Create an agent with summary buffer memory using LCEL pattern.
    
    Args:
        max_token_limit: Maximum tokens before summarizing older messages
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
    
    # Wrap with message history (summary buffer history)
    def get_history(session_id: str) -> BaseChatMessageHistory:
        return get_session_history(session_id, max_token_limit)
    
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    return chain_with_history

def demonstrate_summary_buffer_memory():
    """Demonstrate summary buffer memory using LCEL pattern."""
    print("=" * 60)
    print("Technique 4: Conversation Summary Buffer Memory (LCEL Pattern)")
    print("=" * 60)
    print("Max Token Limit: 200 (will summarize when exceeded)")
    print("Using modern LangChain v1.0+ patterns with RunnableWithMessageHistory")
    print()
    
    chain = create_summary_buffer_memory_agent(max_token_limit=200)
    session_id = "demo_session"
    config = {"configurable": {"session_id": session_id}}
    
    # Simulate a longer conversation
    conversations = [
        "Hi, I'm Diana",
        "I'm a doctor specializing in cardiology",
        "I've been practicing for 15 years",
        "I work at City Hospital",
        "I have two children, Emma and James",
        "My favorite hobby is reading medical journals",
        "What's my name?",
        "Where do I work?",
        "What are my children's names?"
    ]
    
    total_input_tokens = 0
    total_output_tokens = 0
    
    for i, user_input in enumerate(conversations, 1):
        print(f"User: {user_input}")
        
        # Count input tokens (user message + history/summary)
        input_tokens = count_tokens(user_input)
        history = get_session_history(session_id, max_token_limit=200)
        if history.messages:
            input_tokens += count_messages_tokens(history.messages)
        # Add summary tokens if exists
        if hasattr(history, 'summary') and history.summary:
            input_tokens += count_tokens(history.summary)
        total_input_tokens += input_tokens
        
        response = chain.invoke(
            {"input": user_input},
            config=config
        )
        print(f"Agent: {response.content}")
        
        # Count output tokens
        output_tokens = count_tokens(response.content)
        total_output_tokens += output_tokens
        
        # Count current memory tokens (summary + messages)
        history = get_session_history(session_id, max_token_limit=200)
        memory_tokens = count_messages_tokens(history.messages) if history.messages else 0
        if hasattr(history, 'summary') and history.summary:
            memory_tokens += count_tokens(history.summary)
        
        print_token_stats(input_tokens, output_tokens, memory_tokens)
        print()
    
    # Show the stored memory
    print("\n" + "-" * 60)
    print("Stored Memory (Summary + Recent Messages):")
    print("-" * 60)
    history = get_session_history(session_id, max_token_limit=200)
    if hasattr(history, 'summary') and history.summary:
        print("Moving Summary:", history.summary)
    else:
        print("Moving Summary: (not yet created)")
    print("\nRecent Messages:")
    if history.messages:
        for msg in history.messages:
            if isinstance(msg, HumanMessage):
                print(f"  Human: {msg.content[:80]}...")
            elif isinstance(msg, AIMessage):
                print(f"  AI: {msg.content[:80]}...")
    else:
        print("  (no recent messages)")
    print()
    
    # Show total token usage
    final_memory = count_messages_tokens(history.messages) if history.messages else 0
    if hasattr(history, 'summary') and history.summary:
        final_memory += count_tokens(history.summary)
    
    print_token_summary(
        total_input_tokens, 
        total_output_tokens, 
        final_memory
    )

if __name__ == "__main__":
    demonstrate_summary_buffer_memory()

