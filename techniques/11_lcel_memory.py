"""
Technique 11: LCEL (LangChain Expression Language) with Message History
=========================================================================

This technique uses LangChain v1.0's modern LCEL approach with 
RunnableWithMessageHistory. This is the recommended pattern for LangChain v1.0+.

Pros:
- Modern LangChain v1.0 pattern (recommended approach)
- More flexible and composable
- Better integration with LangGraph
- Cleaner API with LCEL
- Better type safety

Cons:
- Different from older ConversationChain pattern
- Requires understanding of LCEL
- Slightly more setup

Use Case: New projects using LangChain v1.0+ where you want to use
the latest recommended patterns.
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

def create_lcel_memory_agent():
    """Create an agent using LCEL with message history (LangChain v1.0+ pattern)."""
    
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
    
    # Wrap with message history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    return chain_with_history

def demonstrate_lcel_memory():
    """Demonstrate LCEL memory pattern."""
    print("=" * 60)
    print("Technique 11: LCEL with Message History (LangChain v1.0+)")
    print("=" * 60)
    print("This is the modern, recommended approach for LangChain v1.0+")
    print()
    
    chain = create_lcel_memory_agent()
    session_id = "demo_session"
    
    # Simulate a conversation
    conversations = [
        "Hi, my name is Julia",
        "I'm a data scientist",
        "I work with Python and machine learning",
        "What's my name?",
        "What do I do for work?",
        "What tools do I use?"
    ]
    
    config = {"configurable": {"session_id": session_id}}
    
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
    
    # Show the stored history
    print("\n" + "-" * 60)
    print("Stored Message History:")
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
    demonstrate_lcel_memory()

