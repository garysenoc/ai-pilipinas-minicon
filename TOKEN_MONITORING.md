# Token Monitoring Guide

All techniques in this project now include token counting to help you monitor and understand token usage across different memory implementations.

## What's Included

### Token Counting Features

1. **Per-Interaction Token Counts**: Shows tokens for each user input and AI response
2. **Memory Token Counts**: Displays current memory size in tokens
3. **Total Token Summary**: Provides cumulative statistics at the end

### Example Output

```
User: Hi, my name is Alice
Agent: Hello Alice! Nice to meet you.
📊 Tokens - Input: 12, Output: 15, Memory: 0

User: What's my name?
Agent: Your name is Alice.
📊 Tokens - Input: 8, Output: 7, Memory: 27

...

------------------------------------------------------------
📊 Total Token Usage Summary:
------------------------------------------------------------
Total Input Tokens:  45
Total Output Tokens: 38
Total Tokens:        83
Final Memory Tokens: 27
```

## Understanding Token Counts

### Input Tokens
- User message tokens
- Memory/history tokens (varies by technique)
- System prompt tokens (estimated)

### Output Tokens
- AI response tokens

### Memory Tokens
- Current size of stored conversation history
- Format depends on memory type:
  - **Buffer Memory**: All messages
  - **Summary Memory**: Compressed summary
  - **Window Memory**: Only recent messages
  - **Entity Memory**: Entities + facts
  - **Vector Store**: Estimated based on stored documents

## Token Counting Implementation

### Utility Module
Located in `utils/token_counter.py`:
- `count_tokens()`: Count tokens in text strings
- `count_messages_tokens()`: Count tokens in LangChain messages
- `count_memory_tokens()`: Count tokens in memory objects
- `format_token_count()`: Format numbers (K/M suffixes)
- `print_token_stats()`: Print formatted statistics
- `print_token_summary()`: Print summary at end

### How It Works

Uses `tiktoken` library to count tokens using the same encoding as GPT models:
- Default: `gpt-3.5-turbo` encoding (cl100k_base)
- Automatically handles different model encodings

## Comparing Techniques

Token monitoring helps you compare different memory techniques:

| Technique | Memory Growth | Token Efficiency |
|-----------|--------------|------------------|
| Buffer Memory | Linear (all messages) | Low |
| Summary Memory | Compressed (summary) | High |
| Window Memory | Fixed (sliding window) | Medium |
| Summary Buffer | Hybrid (summary + recent) | High |
| Entity Memory | Structured (entities only) | Medium |
| LCEL Memory | Depends on backend | Varies |

## Cost Estimation

To estimate costs:
1. Check "Total Tokens" from the summary
2. Multiply by your model's cost per token
3. Example: 1000 tokens × $0.002/1K tokens = $0.002

### Typical Costs (GPT-3.5-turbo)
- Input: ~$0.0015 per 1K tokens
- Output: ~$0.002 per 1K tokens

## Tips for Optimization

1. **Monitor Memory Growth**: Watch how memory tokens increase
2. **Compare Techniques**: Run same conversation with different techniques
3. **Tune Parameters**: Adjust window sizes, token limits, etc.
4. **Use Summarization**: Summary memory reduces token usage significantly
5. **Set Limits**: Use window or summary buffer for long conversations

## Notes

- Token counts are estimates and may vary slightly from actual API usage
- Memory token counts include message structure overhead (~4 tokens per message)
- Vector store token counts are rough estimates
- Some techniques (like summary memory) use additional tokens for summarization (not shown in main counts)

## Troubleshooting

**No token counts showing?**
- Make sure `tiktoken` is installed: `pip install tiktoken`
- Check that the utils module is accessible

**Inaccurate counts?**
- Token counting uses tiktoken which matches OpenAI's tokenizer
- Small differences may occur due to message formatting
- Actual API usage may include additional overhead

## References

- [tiktoken Documentation](https://github.com/openai/tiktoken)
- [OpenAI Token Counting](https://platform.openai.com/tokenizer)
- [LangChain Token Usage](https://python.langchain.com/docs/guides/token_usage)

