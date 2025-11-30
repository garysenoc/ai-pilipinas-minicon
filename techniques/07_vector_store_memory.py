"""
Technique 7: Vector Store Memory (LCEL Pattern)
===============================================

This technique stores conversation history in a vector database, allowing
for semantic search and retrieval of relevant past conversations. It's
useful when you have many conversations and need to find relevant context.
Uses modern LCEL pattern.

Pros:
- Semantic search capabilities
- Can retrieve relevant past conversations
- Scales to many conversations
- Good for long-term memory
- Uses modern LangChain v1.0+ patterns (no deprecation warnings)

Cons:
- Requires vector database setup
- Retrieval may not be perfect
- More complex than simple memory types
- Requires embedding model

Use Case: Applications with many users/conversations where you need to
retrieve relevant past context based on semantic similarity.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import sys
from typing import Dict, List

# Add parent directory to path for utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.token_counter import (
    count_tokens, 
    count_messages_tokens,
    print_token_stats,
    print_token_summary
)

load_dotenv()

# Store for chat message histories and vector stores
store: Dict[str, BaseChatMessageHistory] = {}
vectorstore_store: Dict[str, FAISS] = {}  # session_id -> vectorstore

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat message history for a session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_vectorstore(session_id: str, embeddings: OpenAIEmbeddings) -> FAISS:
    """Get or create vector store for a session."""
    if session_id not in vectorstore_store:
        # Create a new vector store for this session
        vectorstore_store[session_id] = FAISS.from_texts(
            texts=["Initial conversation"],
            embedding=embeddings
        )
    return vectorstore_store[session_id]

def create_vector_store_memory_agent():
    """Create an agent with vector store memory using LCEL pattern."""
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create a custom Runnable that handles vector store retrieval
    class VectorStoreMemoryChain(Runnable):
        """Custom Runnable that handles vector store retrieval with message history."""
        
        def invoke(self, inputs: Dict, config: Dict = None):
            """Invoke with vector store retrieval and message history."""
            session_id = config.get("configurable", {}).get("session_id", "default") if config else "default"
            user_input = inputs.get("input", "")
            history = get_session_history(session_id)
            vectorstore = get_vectorstore(session_id, embeddings)
            
            # Create retriever
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # Retrieve relevant documents
            try:
                # Use invoke for newer LangChain versions
                docs = retriever.invoke(user_input)
            except:
                # Fallback to older API
                try:
                    docs = retriever.get_relevant_documents(user_input)
                except:
                    docs = []
            
            # Store the current conversation in vector store (after getting response)
            # This will be done after the response is generated
            
            # Format retrieved context
            if docs:
                context = "Relevant past conversation context:\n"
                for i, doc in enumerate(docs, 1):
                    context += f"{i}. {doc.page_content}\n"
            else:
                context = "No relevant past conversation context available."
            
            # Create prompt with retrieved context
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a helpful AI assistant. You have access to relevant past conversation context retrieved from memory.

{context}

Use this information to provide accurate and personalized responses."""),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            
            # Invoke prompt and LLM
            messages = prompt.invoke({
                "history": history.messages if history.messages else [],
                "input": user_input
            })
            response = llm.invoke(messages)
            
            # Store the conversation exchange in vector store
            conversation_text = f"Human: {user_input}\nAI: {response.content}"
            if conversation_text.strip() and conversation_text != "Human: \nAI: ":
                try:
                    # Add to vector store
                    vectorstore.add_texts([conversation_text])
                except:
                    pass
            
            return response
    
    # Create the custom chain
    custom_chain = VectorStoreMemoryChain()
    
    # Wrap with message history
    chain_with_history = RunnableWithMessageHistory(
        custom_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    return chain_with_history

def demonstrate_vector_store_memory():
    """Demonstrate vector store memory using LCEL pattern."""
    print("=" * 60)
    print("Technique 7: Vector Store Memory (LCEL Pattern)")
    print("=" * 60)
    print("Using modern LangChain v1.0+ patterns with RunnableWithMessageHistory")
    print()
    
    chain = create_vector_store_memory_agent()
    session_id = "demo_session"
    config = {"configurable": {"session_id": session_id}}
    
    # Initialize embeddings for vector store access
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Simulate a conversation
    conversations = [
        "I love Python programming",
        "I'm working on a machine learning project",
        "I use TensorFlow for deep learning",
        "What programming language do I like?",
        "What am I working on?"
    ]
    
    total_input_tokens = 0
    total_output_tokens = 0
    
    for i, user_input in enumerate(conversations, 1):
        print(f"User: {user_input}")
        
        # Count input tokens (user message + history + retrieved context)
        input_tokens = count_tokens(user_input)
        history = get_session_history(session_id)
        if history.messages:
            input_tokens += count_messages_tokens(history.messages)
        
        # Add retrieved context tokens
        vectorstore = get_vectorstore(session_id, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        try:
            docs = retriever.invoke(user_input)
        except:
            try:
                docs = retriever.get_relevant_documents(user_input)
            except:
                docs = []
        
        for doc in docs:
            input_tokens += count_tokens(doc.page_content)
        total_input_tokens += input_tokens
        
        response = chain.invoke(
            {"input": user_input},
            config=config
        )
        print(f"Agent: {response.content}")
        
        # Count output tokens
        output_tokens = count_tokens(response.content)
        total_output_tokens += output_tokens
        
        # Count current memory tokens (vector store size)
        vectorstore = get_vectorstore(session_id, embeddings)
        memory_tokens = len(vectorstore.index_to_docstore_id) * 50  # Rough estimate
        
        print_token_stats(input_tokens, output_tokens, memory_tokens)
        print()
    
    # Show retrieved documents
    print("\n" + "-" * 60)
    print("Vector Store Statistics:")
    print("-" * 60)
    vectorstore = get_vectorstore(session_id, embeddings)
    print(f"Total documents in vector store: {len(vectorstore.index_to_docstore_id)}")
    print("\nSample retrieval for 'programming':")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    try:
        docs = retriever.invoke("programming")
    except:
        try:
            docs = retriever.get_relevant_documents("programming")
        except:
            docs = []
    
    for doc in docs:
        print(f"  - {doc.page_content[:100]}...")
    print()
    
    # Show total token usage
    final_memory = len(vectorstore.index_to_docstore_id) * 50  # Rough estimate
    print_token_summary(
        total_input_tokens, 
        total_output_tokens, 
        final_memory
    )

if __name__ == "__main__":
    demonstrate_vector_store_memory()
