"""
Custom chat message history implementations for LCEL patterns.

These implement summarization, windowing, and other memory behaviors
that were previously handled by ConversationChain memory classes.
"""

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from typing import List, Optional
from datetime import datetime

class SummaryChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that maintains a running summary."""
    
    def __init__(self, summary_llm: ChatOpenAI):
        self.summary_llm = summary_llm
        self._messages: List[BaseMessage] = []
        self.summary: str = ""
        self.summary_threshold: int = 5  # Summarize after N messages
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Return messages, using summary for older messages."""
        if len(self._messages) <= self.summary_threshold:
            return self._messages
        
        # Return summary + recent messages
        if self.summary:
            return [HumanMessage(content=f"Summary of previous conversation: {self.summary}")] + self._messages[-self.summary_threshold:]
        return self._messages[-self.summary_threshold:]
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message and summarize if needed."""
        self._messages.append(message)
        
        # Summarize when threshold is reached
        if len(self._messages) > self.summary_threshold and len(self._messages) % self.summary_threshold == 0:
            self._summarize()
    
    def _summarize(self) -> None:
        """Summarize older messages."""
        if len(self._messages) <= self.summary_threshold:
            return
        
        # Get messages to summarize (all except recent ones)
        messages_to_summarize = self._messages[:-self.summary_threshold]
        conversation_text = "\n".join([
            f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in messages_to_summarize
        ])
        
        prompt = f"""Summarize the following conversation concisely:

{conversation_text}

Summary:"""
        
        try:
            response = self.summary_llm.invoke(prompt)
            self.summary = response.content if hasattr(response, 'content') else str(response)
            # Keep only recent messages
            self._messages = self._messages[-self.summary_threshold:]
        except:
            pass
    
    def clear(self) -> None:
        """Clear all messages and summary."""
        self._messages = []
        self.summary = ""

class WindowChatMessageHistory(BaseChatMessageHistory):
    """Chat message history with a sliding window."""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._messages: List[BaseMessage] = []
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Return only the most recent messages within the window."""
        return self._messages[-self.window_size:]
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message, discarding older ones if window is exceeded."""
        self._messages.append(message)
        # Keep only last window_size messages
        if len(self._messages) > self.window_size:
            self._messages = self._messages[-self.window_size:]
    
    def clear(self) -> None:
        """Clear all messages."""
        self._messages = []

class SummaryBufferChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that combines summary and recent messages."""
    
    def __init__(self, summary_llm: ChatOpenAI, max_token_limit: int = 200):
        self.summary_llm = summary_llm
        self.max_token_limit = max_token_limit
        self._messages: List[BaseMessage] = []
        self.summary: str = ""
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Return summary message + recent messages."""
        # Estimate tokens and summarize if needed
        self._check_and_summarize()
        
        result = []
        if self.summary:
            result.append(HumanMessage(content=f"Summary: {self.summary}"))
        result.extend(self._messages)
        return result
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message."""
        self._messages.append(message)
        self._check_and_summarize()
    
    def _check_and_summarize(self) -> None:
        """Check token limit and summarize if needed."""
        # Rough token estimation (4 chars ≈ 1 token)
        total_chars = sum(len(str(m.content)) for m in self._messages)
        estimated_tokens = total_chars // 4
        
        if estimated_tokens > self.max_token_limit and len(self._messages) > 2:
            # Summarize older messages, keep recent ones
            keep_recent = 2
            to_summarize = self._messages[:-keep_recent]
            recent = self._messages[-keep_recent:]
            
            conversation_text = "\n".join([
                f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in to_summarize
            ])
            
            prompt = f"""Summarize the following conversation:

{conversation_text}

Summary:"""
            
            try:
                response = self.summary_llm.invoke(prompt)
                new_summary = response.content if hasattr(response, 'content') else str(response)
                # Combine with existing summary
                if self.summary:
                    self.summary = f"{self.summary} {new_summary}"
                else:
                    self.summary = new_summary
                self._messages = recent
            except:
                pass
    
    def clear(self) -> None:
        """Clear all messages and summary."""
        self._messages = []
        self.summary = ""

