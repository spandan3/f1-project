# F1 Chatbot Quick Start

## 1. Build the Database (One-time setup)

```bash
# Convert Parquet files to SQLite
python -m backend.chatbot.db_loader
```

This creates `data/f1_chatbot.db` from your existing Parquet files.

## 2. Install Ollama (Optional - for arbitrary questions)

**Note**: Rule-based queries (fastest lap, winner, pole position) work **without** Ollama. Only install if you want to ask arbitrary questions.

```bash
# macOS: Install via Homebrew (recommended)
brew install ollama

# OR download from https://ollama.ai/

# Then pull the model:
ollama pull llama3.1:8b
```

**Having trouble?** See `backend/chatbot/INSTALL_OLLAMA.md` for detailed installation instructions.

## 3. Test the Chatbot

```python
from backend.chatbot import ask_question

# Rule-based (fast, no LLM needed)
response = ask_question("Who had the fastest lap in Monaco 2023?", use_llm=False)
print(response["answer"])

# With LLM fallback (for arbitrary questions)
response = ask_question("Show me all races where Verstappen finished in the top 3 in 2023")
print(response["answer"])
print(response["sql"])  # See the generated SQL
```

## 4. Use via API

```bash
# Start the API
python run_api.py

# Query the chatbot
curl "http://localhost:8000/chat?question=Who%20won%20Bahrain%202024"
```

## Example Questions

### Rule-based (fast, no Ollama needed):
- "Who had the fastest lap in Monaco 2023?"
- "Who won the Bahrain Grand Prix in 2024?"
- "Who got pole position in Monaco 2023?"

### LLM-based (requires Ollama):
- "Show me all races where Hamilton got pole in 2023"
- "Which driver had the most fastest laps in 2024?"

## Response Format

```python
{
    "answer": "Max Verstappen had the fastest lap with a time of 1:15.650.",
    "rows": [{"Driver": "Max Verstappen", ...}],
    "explanation": "I executed a SQL query...",
    "sql": "SELECT Driver, MIN(LapTime) ...",
    "method": "rule_based",  # or "llm"
    "row_count": 1,
    "error": None
}
```

## Troubleshooting

**"Database not found"**
```bash
python -m backend.chatbot.db_loader
```

**"Ollama not found"** (only needed for arbitrary questions)
```bash
# Install from https://ollama.ai/
ollama pull llama3.1:8b
```

**"No data found"**
- Check that Parquet files exist in `data/raw/`
- Verify the year/race exists in your dataset
- Check `response["sql"]` to see what query was generated

