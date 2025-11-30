"""
Technique 2: Conversation Summary Memory (LCEL Pattern)
=======================================================

This technique maintains a running summary of the conversation instead of
storing all messages. It uses an LLM to summarize the conversation history
periodically. Uses modern LCEL pattern.

Pros:
- Efficient for long conversations (reduces token usage)
- Automatically compresses information
- Can handle very long conversation histories
- Uses modern LangChain v1.0+ patterns (no deprecation warnings)

Cons:
- Some detail may be lost in summarization
- Requires additional LLM calls for summarization
- Summary quality depends on the summarization prompt

Use Case: Long-running conversations where you need to maintain context
but want to reduce token costs.
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
from utils.custom_chat_histories import SummaryChatMessageHistory

load_dotenv()

# Store for chat message histories
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create summary chat message history for a session."""
    if session_id not in store:
        summary_llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        store[session_id] = SummaryChatMessageHistory(summary_llm=summary_llm)
    return store[session_id]

def create_summary_memory_agent():
    """Create an agent with summary memory using LCEL pattern."""
    
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
    
    # Wrap with message history (summary history)
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    return chain_with_history

def demonstrate_summary_memory():
    """Demonstrate summary memory using LCEL pattern."""
    print("=" * 60)
    print("Technique 2: Conversation Summary Memory (LCEL Pattern)")
    print("=" * 60)
    print("Using modern LangChain v1.0+ patterns with RunnableWithMessageHistory")
    print()
    
    chain = create_summary_memory_agent()
    session_id = "demo_session"
    config = {"configurable": {"session_id": session_id}}
    
    # Simulate a longer conversation
    conversations = [
        "Hi, I'm Bob and I work as a data scientist",
        "I specialize in machine learning and deep learning",
        "I've been working on NLP projects for 5 years",
        "My favorite programming language is Python",
        "What's my profession?",
        "What programming language do I prefer?",
        "How long have I been working on NLP?"
    ]
    
    total_input_tokens = 0
    total_output_tokens = 0
    
    for i, user_input in enumerate(conversations, 1):
        print(f"User: {user_input}")
        
        # Count input tokens (user message + history/summary)
        input_tokens = count_tokens(user_input)
        history = get_session_history(session_id)
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
        history = get_session_history(session_id)
        memory_tokens = count_messages_tokens(history.messages) if history.messages else 0
        if hasattr(history, 'summary') and history.summary:
            memory_tokens += count_tokens(history.summary)
        
        print_token_stats(input_tokens, output_tokens, memory_tokens)
        print()
    
    # Show the stored summary
    print("\n" + "-" * 60)
    print("Stored Summary:")
    print("-" * 60)
    history = get_session_history(session_id)
    if hasattr(history, 'summary') and history.summary:
        print(history.summary)
    else:
        print("(Summary will be created after more messages)")
    print(f"\nRecent Messages: {len(history.messages)}")
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
    demonstrate_summary_memory()

