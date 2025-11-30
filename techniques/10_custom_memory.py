"""
Technique 10: Custom Memory Implementation
===========================================

This demonstrates how to create a custom memory class that implements
specific behavior. You can combine multiple techniques or implement
domain-specific logic.

Pros:
- Full control over memory behavior
- Can implement custom logic
- Can combine multiple techniques
- Domain-specific optimizations

Cons:
- More development effort
- Need to maintain custom code
- Must handle edge cases

Use Case: When you need specific memory behavior that isn't available
in standard LangChain memory types, or want to combine multiple approaches.
"""

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv
import os
import sys
from datetime import datetime

# Add parent directory to path for utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.token_counter import (
    count_tokens, 
    count_messages_tokens,
    print_token_stats,
    print_token_summary
)

load_dotenv()

class TimestampedChatMessageHistory(BaseChatMessageHistory):
    """Custom chat message history that stores messages with timestamps and allows
    time-based filtering."""
    
    def __init__(self, max_age_seconds: int = 3600):
        self.max_age_seconds = max_age_seconds
        self.messages_with_timestamps: List[tuple] = []  # (message, timestamp)
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Return messages filtered by age."""
        current_time = datetime.now()
        return [
            msg for msg, timestamp in self.messages_with_timestamps
            if (current_time - timestamp).total_seconds() < self.max_age_seconds
        ]
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message with timestamp."""
        current_time = datetime.now()
        self.messages_with_timestamps.append((message, current_time))
    
    def clear(self) -> None:
        """Clear all messages."""
        self.messages_with_timestamps = []
    
    def get_all_messages(self) -> List[BaseMessage]:
        """Get all messages (including expired ones) for inspection."""
        return [msg for msg, _ in self.messages_with_timestamps]

# Store for custom chat message histories
custom_store: Dict[str, TimestampedChatMessageHistory] = {}

def get_custom_session_history(session_id: str, max_age_seconds: int = 3600) -> BaseChatMessageHistory:
    """Get or create custom timestamped chat message history for a session."""
    if session_id not in custom_store:
        custom_store[session_id] = TimestampedChatMessageHistory(max_age_seconds=max_age_seconds)
    return custom_store[session_id]

def create_custom_memory_agent(max_age_seconds=3600):
    """Create an agent with custom timestamped memory using LCEL pattern.
    
    Args:
        max_age_seconds: Maximum age of messages to keep (default 1 hour)
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
    
    # Wrap with custom message history (time-based filtering)
    def get_history(session_id: str) -> BaseChatMessageHistory:
        return get_custom_session_history(session_id, max_age_seconds)
    
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    return chain_with_history

def demonstrate_custom_memory():
    """Demonstrate custom memory using LCEL pattern."""
    print("=" * 60)
    print("Technique 10: Custom Memory Implementation (LCEL Pattern)")
    print("=" * 60)
    print("Custom Feature: Time-based message filtering")
    print("Max Age: 3600 seconds (1 hour)")
    print("Using modern LangChain v1.0+ patterns")
    print()
    
    chain = create_custom_memory_agent(max_age_seconds=3600)
    session_id = "demo_session"
    config = {"configurable": {"session_id": session_id}}
    
    # Simulate a conversation
    conversations = [
        "Hi, I'm Ivy",
        "I'm a UX designer",
        "I specialize in mobile app design",
        "What's my name?",
        "What do I do?"
    ]
    
    total_input_tokens = 0
    total_output_tokens = 0
    
    for i, user_input in enumerate(conversations, 1):
        print(f"User: {user_input}")
        
        # Count input tokens (user message + filtered history)
        input_tokens = count_tokens(user_input)
        history = get_custom_session_history(session_id, max_age_seconds=3600)
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
        
        # Count current memory tokens (filtered by age)
        history = get_custom_session_history(session_id, max_age_seconds=3600)
        memory_tokens = count_messages_tokens(history.messages) if history.messages else 0
        
        print_token_stats(input_tokens, output_tokens, memory_tokens)
        print()
    
    # Show the stored memory with timestamps
    print("\n" + "-" * 60)
    print("Stored Memory (with Timestamps - All Messages):")
    print("-" * 60)
    history = get_custom_session_history(session_id, max_age_seconds=3600)
    all_messages = history.get_all_messages()
    for msg, timestamp in history.messages_with_timestamps:
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {msg.content[:50]}...")
    print()
    
    # Show total token usage
    final_memory = count_messages_tokens(history.messages) if history.messages else 0
    print_token_summary(
        total_input_tokens, 
        total_output_tokens, 
        final_memory
    )

if __name__ == "__main__":
    demonstrate_custom_memory()

