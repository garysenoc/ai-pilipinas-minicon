"""
Utility module for counting tokens in LangChain messages and strings.

This helps monitor token usage across different memory techniques.
"""

from typing import List, Union
from langchain_core.messages import BaseMessage
import tiktoken

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in a text string.
    
    Args:
        text: The text to count tokens for
        model: The model name (determines encoding)
    
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback to cl100k_base encoding (used by gpt-3.5-turbo and gpt-4)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

def count_messages_tokens(messages: List[BaseMessage], model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in a list of LangChain messages.
    
    Args:
        messages: List of LangChain message objects
        model: The model name (determines encoding)
    
    Returns:
        Total number of tokens
    """
    total_tokens = 0
    for message in messages:
        # Count content tokens
        if hasattr(message, 'content'):
            content = str(message.content)
            total_tokens += count_tokens(content, model)
        
        # Add overhead for message structure (role, etc.)
        # Rough estimate: ~4 tokens per message for structure
        total_tokens += 4
    
    return total_tokens

def count_memory_tokens(memory, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in a memory object.
    
    Args:
        memory: LangChain memory object
        model: The model name (determines encoding)
    
    Returns:
        Total number of tokens
    """
    tokens = 0
    
    # Try different memory attributes
    if hasattr(memory, 'buffer'):
        if isinstance(memory.buffer, str):
            tokens += count_tokens(memory.buffer, model)
        elif isinstance(memory.buffer, list):
            tokens += count_messages_tokens(memory.buffer, model)
    
    if hasattr(memory, 'chat_memory'):
        if hasattr(memory.chat_memory, 'messages'):
            tokens += count_messages_tokens(memory.chat_memory.messages, model)
    
    if hasattr(memory, 'moving_summary'):
        if memory.moving_summary:
            tokens += count_tokens(memory.moving_summary, model)
    
    if hasattr(memory, 'entity_store'):
        if hasattr(memory.entity_store, 'store'):
            # Count entity facts
            for entity, facts in memory.entity_store.store.items():
                tokens += count_tokens(entity, model)
                for fact in facts:
                    tokens += count_tokens(fact, model)
    
    return tokens

def format_token_count(tokens: int) -> str:
    """
    Format token count with K/M suffixes for readability.
    
    Args:
        tokens: Number of tokens
    
    Returns:
        Formatted string
    """
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.2f}M"
    elif tokens >= 1_000:
        return f"{tokens / 1_000:.2f}K"
    else:
        return str(tokens)

def print_token_stats(input_tokens: int, output_tokens: int, memory_tokens: int = None):
    """
    Print formatted token statistics.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        memory_tokens: Number of tokens in memory (optional)
    """
    stats = f"📊 Tokens - Input: {format_token_count(input_tokens)}, Output: {format_token_count(output_tokens)}"
    if memory_tokens is not None:
        stats += f", Memory: {format_token_count(memory_tokens)}"
    print(stats)

def print_token_summary(total_input: int, total_output: int, final_memory: int = None):
    """
    Print total token usage summary.
    
    Args:
        total_input: Total input tokens
        total_output: Total output tokens
        final_memory: Final memory token count (optional)
    """
    print("-" * 60)
    print("📊 Total Token Usage Summary:")
    print("-" * 60)
    print(f"Total Input Tokens:  {format_token_count(total_input)}")
    print(f"Total Output Tokens: {format_token_count(total_output)}")
    print(f"Total Tokens:        {format_token_count(total_input + total_output)}")
    if final_memory is not None:
        print(f"Final Memory Tokens: {format_token_count(final_memory)}")
    print()

