# LangChain Version Compatibility

## Current Status

This project is **compatible with LangChain v1.0+** and has been updated to align with the latest practices from [langchain.com](http://langchain.com).

## What's Included

### Traditional Patterns (Techniques 1-10)
These techniques use the `ConversationChain` pattern which is still supported in LangChain v1.0:
- ✅ `ConversationBufferMemory`
- ✅ `ConversationSummaryMemory`
- ✅ `ConversationBufferWindowMemory`
- ✅ `ConversationSummaryBufferMemory`
- ✅ `ConversationEntityMemory`
- ✅ `ConversationKGMemory`
- ✅ `VectorStoreRetrieverMemory`
- ✅ Redis-backed memory
- ✅ SQL-backed memory
- ✅ Custom memory implementations

### Modern Patterns (Technique 11)
**Technique 11** uses the modern LCEL (LangChain Expression Language) pattern with `RunnableWithMessageHistory`, which is the **recommended approach** for LangChain v1.0+:

- ✅ `RunnableWithMessageHistory`
- ✅ LCEL pipe operator (`|`)
- ✅ `ChatPromptTemplate` with `MessagesPlaceholder`
- ✅ `BaseChatMessageHistory` interface

## Version Requirements

- **LangChain**: >= 0.3.0 (supports v1.0+)
- **LangChain OpenAI**: >= 0.2.0
- **LangChain Community**: >= 0.3.0

## Migration Guide

If you're using older LangChain versions (< 0.3.0), you may need to:

1. **Update dependencies**:
   ```bash
   pip install --upgrade langchain langchain-openai langchain-community
   ```

2. **For new projects**: Use Technique 11 (LCEL) as it's the recommended pattern

3. **For existing projects**: Techniques 1-10 will continue to work, but consider migrating to LCEL for better future compatibility

## Key Changes in LangChain v1.0

1. **LCEL is the recommended pattern** for building chains
2. **RunnableWithMessageHistory** is the modern way to handle chat history
3. **ChatMessageHistory** classes are in `langchain_community`
4. **Core abstractions** moved to `langchain_core`

## References

- [LangChain Chat History Documentation](https://python.langchain.com/docs/concepts/chat_history/)
- [LangChain LCEL Documentation](https://python.langchain.com/docs/expression_language/)
- [LangChain Official Site](http://langchain.com)

## Support

All techniques in this project have been verified to work with LangChain v1.0+. If you encounter any compatibility issues, please check:

1. Your LangChain version: `pip show langchain`
2. The specific technique's imports
3. The official LangChain documentation for any breaking changes

