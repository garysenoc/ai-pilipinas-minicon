"""
Technique 5: Conversation Entity Memory (LCEL Pattern)
========================================================

This technique extracts and stores entities (people, places, things) from
the conversation separately, allowing the agent to remember specific facts
about entities across the conversation. Uses modern LCEL pattern.

Pros:
- Explicitly tracks important entities and facts
- Good for remembering user preferences and details
- Structured memory format
- Can query specific entities
- Uses modern LangChain v1.0+ patterns (no deprecation warnings)

Cons:
- Only stores entity-related information
- May miss non-entity context
- Requires entity extraction (can be imperfect)

Use Case: Applications where you need to remember specific facts about
users, products, or other entities (e.g., personal assistants, CRM systems).
"""

from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, Runnable
from dotenv import load_dotenv
import os
import sys
from typing import Dict, List
import json
import re

# Add parent directory to path for utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.token_counter import (
    count_tokens, 
    count_messages_tokens,
    print_token_stats,
    print_token_summary
)

load_dotenv()

# Store for chat message histories and entities
store: Dict[str, BaseChatMessageHistory] = {}
entity_store: Dict[str, Dict[str, List[str]]] = {}  # session_id -> {entity: [facts]}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat message history for a session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def extract_entities_from_conversation(llm: ChatOpenAI, session_id: str, messages: List, user_input: str) -> Dict[str, List[str]]:
    """Extract entities and facts from the conversation using LLM."""
    if session_id not in entity_store:
        entity_store[session_id] = {}
    
    # Build conversation context
    conversation_text = "\n".join([
        f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in messages[-5:]  # Last 5 messages for context
    ])
    
    extraction_prompt = f"""Extract entities and facts from this conversation. Return a JSON object with entities as keys and lists of facts as values.

Conversation:
{conversation_text}
Human: {user_input}

Return only valid JSON in this format: {{"EntityName": ["fact1", "fact2"], ...}}

Example:
{{"Frank": ["30 years old", "lives in New York"], "New York": ["city where Frank lives"]}}"""

    try:
        response = llm.invoke(extraction_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            entities = json.loads(json_match.group())
        else:
            entities = json.loads(content)
        
        # Merge with existing entities
        for entity, facts in entities.items():
            if entity not in entity_store[session_id]:
                entity_store[session_id][entity] = []
            for fact in facts:
                if fact and fact not in entity_store[session_id][entity]:
                    entity_store[session_id][entity].append(fact)
    except Exception as e:
        # If extraction fails, continue without updating entities
        pass
    
    return entity_store.get(session_id, {})

def format_entities(entities: Dict[str, List[str]]) -> str:
    """Format entities for the prompt."""
    if not entities:
        return "No entities tracked yet."
    formatted = []
    for entity, facts in entities.items():
        if facts:
            formatted.append(f"{entity}: {', '.join(facts)}")
        else:
            formatted.append(f"{entity}: (no facts yet)")
    return "\n".join(formatted)

def create_entity_memory_agent():
    """Create an agent with entity memory using LCEL pattern."""
    
    # Initialize the LLM for conversation
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # LLM for entity extraction
    extraction_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    def add_entities_with_context(inputs: Dict, config: Dict = None) -> Dict:
        """Extract and add entity information to the input."""
        # Get session_id from config
        if config:
            session_id = config.get("configurable", {}).get("session_id", "default")
        else:
            session_id = inputs.get("config", {}).get("configurable", {}).get("session_id", "default")
        
        history = get_session_history(session_id)
        user_input = inputs.get("input", "")
        
        # Extract entities from the conversation
        entities = extract_entities_from_conversation(
            extraction_llm, 
            session_id, 
            history.messages if history.messages else [], 
            user_input
        )
        
        # Format entities
        entities_text = format_entities(entities)
        
        # Add entities to input dict
        inputs["entities"] = entities_text
        return inputs
    
    # Create prompt template with entity information
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. You have access to information about entities mentioned in the conversation.

Relevant entity information:
{entities}

Use this information to provide accurate and personalized responses."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Create a chain that extracts entities and processes
    # We'll use RunnableLambda with proper config handling
    def process_with_entities(inputs: Dict, config: Dict = None):
        """Process input with entity extraction."""
        # Get session_id from config (provided by RunnableWithMessageHistory)
        if config:
            session_id = config.get("configurable", {}).get("session_id", "default")
        else:
            session_id = "default"
        
        user_input = inputs.get("input", "")
        history = get_session_history(session_id)
        
        # Extract entities from conversation
        entities = extract_entities_from_conversation(
            extraction_llm,
            session_id,
            history.messages if history.messages else [],
            user_input
        )
        entities_text = format_entities(entities)
        
        # Create prompt input with entities and history
        prompt_input = {
            "input": user_input,
            "entities": entities_text,
            "history": history.messages if history.messages else []
        }
        
        # Invoke prompt and LLM
        messages = prompt.invoke(prompt_input)
        response = llm.invoke(messages)
        return response
    
    # Create RunnableLambda that properly handles config
    entity_chain = RunnableLambda(process_with_entities)
    
    # Wrap with message history
    # RunnableWithMessageHistory will add history, but we need to extract entities
    # So we'll create a custom wrapper
    class EntityMemoryChain(Runnable):
        """Custom Runnable that handles entity extraction with message history."""
        
        def invoke(self, inputs: Dict, config: Dict = None):
            """Invoke with entity extraction and message history."""
            session_id = config.get("configurable", {}).get("session_id", "default") if config else "default"
            user_input = inputs.get("input", "")
            history = get_session_history(session_id)
            
            # Extract entities
            entities = extract_entities_from_conversation(
                extraction_llm,
                session_id,
                history.messages if history.messages else [],
                user_input
            )
            entities_text = format_entities(entities)
            
            # Create prompt input
            prompt_input = {
                "input": user_input,
                "entities": entities_text,
                "history": history.messages if history.messages else []
            }
            
            # Invoke prompt and LLM
            messages = prompt.invoke(prompt_input)
            response = llm.invoke(messages)
            return response
    
    # Create the custom chain
    custom_chain = EntityMemoryChain()
    
    # Wrap with message history
    chain_with_history = RunnableWithMessageHistory(
        custom_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    return chain_with_history

def demonstrate_entity_memory():
    """Demonstrate entity memory using LCEL pattern."""
    print("=" * 60)
    print("Technique 5: Conversation Entity Memory (LCEL Pattern)")
    print("=" * 60)
    print("Using modern LangChain v1.0+ patterns with RunnableWithMessageHistory")
    print()
    
    chain = create_entity_memory_agent()
    session_id = "demo_session"
    config = {"configurable": {"session_id": session_id}}
    
    # Simulate a conversation with entities
    conversations = [
        "Hi, I'm Frank and I'm 30 years old",
        "I live in New York",
        "My favorite color is blue",
        "I work at TechCorp as a developer",
        "What's my name?",
        "How old am I?",
        "Where do I live?",
        "What's my favorite color?",
        "Where do I work?"
    ]
    
    total_input_tokens = 0
    total_output_tokens = 0
    
    for i, user_input in enumerate(conversations, 1):
        print(f"User: {user_input}")
        
        # Count input tokens (user message + history + entities)
        input_tokens = count_tokens(user_input)
        history = get_session_history(session_id)
        if history.messages:
            input_tokens += count_messages_tokens(history.messages)
        # Add entity tokens
        if session_id in entity_store:
            for entity, facts in entity_store[session_id].items():
                input_tokens += count_tokens(entity)
                for fact in facts:
                    input_tokens += count_tokens(fact)
        total_input_tokens += input_tokens
        
        response = chain.invoke(
            {"input": user_input},
            config=config
        )
        print(f"Agent: {response.content}")
        
        # Count output tokens
        output_tokens = count_tokens(response.content)
        total_output_tokens += output_tokens
        
        # Count current memory tokens (history + entities)
        history = get_session_history(session_id)
        memory_tokens = count_messages_tokens(history.messages) if history.messages else 0
        if session_id in entity_store:
            for entity, facts in entity_store[session_id].items():
                memory_tokens += count_tokens(entity)
                for fact in facts:
                    memory_tokens += count_tokens(fact)
        
        print_token_stats(input_tokens, output_tokens, memory_tokens)
        print()
    
    # Show the stored entities
    print("\n" + "-" * 60)
    print("Stored Entities:")
    print("-" * 60)
    if session_id in entity_store:
        entities = entity_store[session_id]
        for entity, facts in entities.items():
            print(f"\nEntity: {entity}")
            for fact in facts:
                print(f"  - {fact}")
    else:
        print("No entities extracted yet.")
    print()
    
    # Show total token usage
    history = get_session_history(session_id)
    final_memory = count_messages_tokens(history.messages) if history.messages else 0
    if session_id in entity_store:
        for entity, facts in entity_store[session_id].items():
            final_memory += count_tokens(entity)
            for fact in facts:
                final_memory += count_tokens(fact)
    
    print_token_summary(
        total_input_tokens, 
        total_output_tokens, 
        final_memory
    )

if __name__ == "__main__":
    demonstrate_entity_memory()

