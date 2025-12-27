# F1 Chatbot: Natural Language Query Interface

A free, local-data chatbot for querying F1 analytics data using natural language questions.

## Architecture Overview

The chatbot uses a **hybrid approach**:

1. **Rule-based handlers** (fast, accurate) for common queries:
   - Fastest lap queries
   - Race winner queries
   - Pole position queries
   - Team/driver points queries

2. **LLM-based NL→SQL translation** (Ollama) for arbitrary questions:
   - Uses local LLM (Llama 3.1:8b) - completely free
   - Converts natural language → SQL
   - Validates SQL for safety (read-only)

3. **SQLite backend**:
   - Converts Parquet files → SQLite database
   - All queries are read-only SELECT statements
   - Safe execution with validation

## Setup

### 1. Install Ollama (for LLM fallback)

```bash
# Install from https://ollama.ai/
# Then pull the model:
ollama pull llama3.1:8b
```

**Note:** The chatbot will work with rule-based handlers even without Ollama, but won't handle arbitrary questions.

### 2. Build SQLite Database

Convert your Parquet files to SQLite:

```bash
# Option 1: Using Python module
python -m backend.chatbot.db_loader

# Option 2: Force rebuild (if database exists)
python -c "from backend.chatbot.db_loader import load_parquet_to_sqlite; load_parquet_to_sqlite(force_rebuild=True)"
```

This creates `data/f1_chatbot.db` with three tables:
- `results`: Race and qualifying results
- `laps`: Lap-by-lap data
- `weather`: Weather data per session

### 3. Verify Setup

```bash
# Test the chatbot
python -m backend.chatbot.chatbot
```

## Usage

### Python API

```python
from backend.chatbot import ask_question

# Ask a question
response = ask_question("Who had the fastest lap in Monaco 2023?")

print(response["answer"])
# "Max Verstappen had the fastest lap with a time of 1:15.650."

print(response["sql"])
# SELECT Driver, MIN(LapTime) as FastestLapTime FROM laps WHERE ...

print(response["rows"])
# [{"Driver": "Max Verstappen", "FastestLapTime": "0 days 01:15.650000", ...}]
```

### REST API

```bash
# Start the API server
python run_api.py

# Query the chatbot
curl "http://localhost:8000/chat?question=Who%20won%20Bahrain%202024"
```

### Response Format

```json
{
  "answer": "Max Verstappen won the race.",
  "rows": [
    {
      "Driver": "Max Verstappen",
      "TeamName": "Red Bull Racing",
      "FinishPosition": 1,
      "event_year": 2024,
      "event_name": "Bahrain Grand Prix"
    }
  ],
  "explanation": "I executed a SQL query against the F1 database...",
  "sql": "SELECT Driver, TeamName, Position as FinishPosition FROM results WHERE ...",
  "method": "rule_based",  // or "llm"
  "row_count": 1,
  "error": null
}
```

## Example Questions

### Rule-based (fast, no LLM needed)

- "Who had the fastest lap in Monaco 2023?"
- "Who won the Bahrain Grand Prix in 2024?"
- "Who got pole position in Monaco 2023?"
- "Which team scored the most points in 2022?"

### LLM-based (requires Ollama)

- "Show me all races where Verstappen finished in the top 3 in 2023"
- "What was the average lap time for Hamilton in Silverstone 2024?"
- "List all drivers who started from the front row in 2023"

## Safety Features

### SQL Validation

All SQL queries are validated before execution:

- ✅ Must start with `SELECT` or `WITH` (CTEs)
- ✅ Blocks dangerous keywords: `INSERT`, `UPDATE`, `DELETE`, `DROP`, etc.
- ✅ Read-only operations only
- ✅ Prevents SQL injection via keyword checking

### Error Handling

- If rule-based handler fails → falls back to LLM
- If LLM fails → returns helpful error message
- If SQL execution fails → returns error with explanation
- If no data found → explains why (year/race not in dataset)

## Database Schema

The chatbot understands these tables:

### `results` table
- `event_year`, `event_name`, `session_type` (Q/R)
- `Driver`, `TeamName`, `DriverNumber`
- `Position`, `GridPosition`
- `Q1`, `Q2`, `Q3` (qualifying times)
- `FastestLap`, `FastestLapTime`, `Points`, `Status`

### `laps` table
- `event_year`, `event_name`, `session_type`
- `Driver`, `TeamName`, `DriverNumber`
- `LapNumber`, `LapTime`, `Position`
- `Sector1Time`, `Sector2Time`, `Sector3Time`
- `Compound`, `TyreLife`, `IsPersonalBest`

### `weather` table
- `event_year`, `event_name`, `session_type`
- `AirTemp`, `TrackTemp`, `Humidity`, `Rainfall`
- `WindSpeed`, `WindDirection`

## How It Works

### Rule-based Handler Flow

```
Question: "Who had the fastest lap in Monaco 2023?"
    ↓
1. Detect intent: "fastest_lap"
2. Extract year: 2023, race: "Monaco Grand Prix"
3. Generate SQL: SELECT Driver, MIN(LapTime) FROM laps WHERE ...
4. Execute & validate SQL
5. Format response
```

### LLM-based Flow

```
Question: "Show me Verstappen's average lap time in all 2023 races"
    ↓
1. Rule-based fails (no matching pattern)
2. Build schema prompt with database structure
3. Call Ollama with NL→SQL prompt
4. Extract SQL from LLM response
5. Validate SQL (safety check)
6. Execute & format response
```

## LLM Prompt Example

The chatbot sends this prompt to Ollama:

```
You are a SQL query generator for Formula 1 race data. Your ONLY job is to 
convert natural language questions into SQL queries.

Database Schema:
[Detailed schema with tables and columns]

Rules:
1. You MUST generate ONLY a SQL SELECT query. Do not include any explanation.
2. The query must be valid SQLite syntax.
3. Use exact column names and table names from the schema above.
4. For event names, use the exact format from the database 
   (e.g., "Monaco Grand Prix" not just "Monaco").
5. For years, use integer comparison (e.g., event_year = 2023).

Example queries:
- "Who had the fastest lap in Monaco 2023?" → 
  SELECT Driver, MIN(LapTime) FROM laps WHERE event_year = 2023 
  AND event_name = 'Monaco Grand Prix' AND session_type = 'R' 
  GROUP BY Driver ORDER BY MIN(LapTime) LIMIT 1

Now generate SQL for this question:
[User's question]

SQL:
```

## Troubleshooting

### "Database not found"

```bash
# Build the database
python -m backend.chatbot.db_loader
```

### "Ollama not found"

```bash
# Install Ollama: https://ollama.ai/
# Pull model:
ollama pull llama3.1:8b
```

### "No data found"

- Check that your Parquet files exist in `data/raw/`
- Verify the year/race exists in your dataset
- Check SQL logs in response["sql"] to debug

### "Failed to generate valid SQL"

- Try rephrasing your question
- Use simpler queries (the LLM works better with straightforward questions)
- Check that column names match the schema

## Customization

### Add Rule-based Handler

Edit `backend/chatbot/rule_handlers.py`:

```python
def handle_custom_intent(question: str) -> Optional[str]:
    year, race_name = extract_year_and_race(question)
    if "your_pattern" in question.lower():
        return f"SELECT ... FROM ... WHERE event_year = {year} ..."
    return None
```

### Change LLM Model

Edit `backend/chatbot/nl_to_sql.py`:

```python
OLLAMA_MODEL = "mistral"  # or "codellama", "llama3", etc.
```

### Customize SQL Validation

Edit `backend/chatbot/sql_validator.py` to add/remove dangerous keywords.

## Performance

- **Rule-based queries**: < 50ms
- **LLM queries**: 1-5 seconds (depends on Ollama model)
- **SQL execution**: < 100ms (SQLite is fast)

## Limitations

1. **Event name matching**: Rule-based handlers require exact race names. LLM queries are more flexible.

2. **Column availability**: Some columns may not exist in older data (e.g., `Points` might be missing).

3. **LLM accuracy**: The LLM may generate incorrect SQL for complex queries. Always check `response["sql"]`.

4. **Local only**: This system is designed for local use. No cloud APIs, completely free.

## Future Improvements

- [ ] Add more rule-based handlers (podium, DNFs, etc.)
- [ ] Support for aggregate queries (average lap times, etc.)
- [ ] Query result caching
- [ ] Better error messages with suggestions
- [ ] Support for date ranges ("all races in 2023-2024")
- [ ] Comparison queries ("compare Verstappen vs Hamilton")

