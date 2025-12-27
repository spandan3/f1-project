# F1 Chatbot: Complete Design & Implementation Guide

This document explains the architecture and design of the F1 data chatbot system.

## Overview

The chatbot is a **free, local-data query system** that answers natural language questions about F1 racing data by:
1. Converting questions to SQL queries
2. Executing queries against a local SQLite database
3. Returning factual answers based solely on the dataset

**Key principle**: The LLM never answers from its own knowledge. It only translates English → SQL. All answers come from the dataset.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Question                            │
│  "Who had the fastest lap in Monaco 2023?"                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   Intent Detection            │
        │   (Rule-based Patterns)       │
        └──────┬───────────────┬────────┘
               │               │
        ┌──────▼─────┐  ┌──────▼─────────┐
        │ Rule-based │  │ LLM (Ollama)   │
        │ Handler    │  │ NL→SQL         │
        └──────┬─────┘  └──────┬─────────┘
               │               │
               └───────┬───────┘
                       ▼
        ┌──────────────────────────────┐
        │   SQL Validator               │
        │   - Read-only check           │
        │   - Keyword blocking          │
        └──────┬───────────────────────┘
               │
               ▼
        ┌──────────────────────────────┐
        │   SQLite Database             │
        │   (Parquet → SQLite)          │
        └──────┬───────────────────────┘
               │
               ▼
        ┌──────────────────────────────┐
        │   Response Formatter          │
        │   - Natural language answer   │
        │   - Source rows               │
        │   - Explanation               │
        └──────────────────────────────┘
```

---

## Components

### 1. Database Loader (`db_loader.py`)

**Purpose**: Convert Parquet files to SQLite for querying.

**Key Functions**:
- `load_parquet_to_sqlite()`: Converts `results.parquet`, `laps.parquet`, `weather.parquet` → SQLite
- `get_db_schema()`: Generates schema description for LLM prompts

**Output**: `data/f1_chatbot.db` with tables:
- `results` (race & qualifying results)
- `laps` (lap-by-lap data)
- `weather` (weather conditions)

**Example**:
```python
from backend.chatbot.db_loader import load_parquet_to_sqlite
db_path = load_parquet_to_sqlite(force_rebuild=False)
```

---

### 2. SQL Validator (`sql_validator.py`)

**Purpose**: Ensure all SQL queries are read-only and safe.

**Validation Rules**:
- ✅ Must start with `SELECT` or `WITH` (CTEs)
- ❌ Blocks: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `CREATE`, `ALTER`, etc.
- ✅ Allows read-only `PRAGMA` (table_info, index_list, etc.)
- ❌ Blocks multiple statements (semicolon injection)

**Example**:
```python
from backend.chatbot.sql_validator import validate_sql

sql = "SELECT * FROM results WHERE event_year = 2023"
is_valid, error = validate_sql(sql)
# (True, None)

sql_dangerous = "DROP TABLE results"
is_valid, error = validate_sql(sql_dangerous)
# (False, "Dangerous keyword detected: DROP")
```

---

### 3. Rule-based Handlers (`rule_handlers.py`)

**Purpose**: Fast, accurate SQL generation for common question types.

**Supported Intents**:
- `fastest_lap`: "Who had the fastest lap in Monaco 2023?"
- `race_winner`: "Who won the Bahrain Grand Prix in 2024?"
- `pole_position`: "Who got pole position in Monaco 2023?"
- `team_points`: "Which team scored the most points in 2022?"
- `driver_points`: "Which driver scored the most points in 2023?"

**How It Works**:
1. Pattern matching (regex) to detect intent
2. Extract year and race name from question
3. Generate SQL template with parameters

**Example SQL for "fastest lap in Monaco 2023"**:
```sql
SELECT 
    Driver,
    TeamName,
    MIN(LapTime) as FastestLapTime,
    event_year,
    event_name
FROM laps
WHERE event_year = 2023
    AND event_name = 'Monaco Grand Prix'
    AND session_type = 'R'
    AND LapTime IS NOT NULL
    AND LapTime != 'NaT'
    AND LapTime != ''
GROUP BY Driver, TeamName, event_year, event_name
ORDER BY FastestLapTime ASC
LIMIT 1
```

**Advantages**:
- Fast (< 50ms)
- 100% accurate (no LLM hallucinations)
- No external dependencies

---

### 4. NL→SQL Translator (`nl_to_sql.py`)

**Purpose**: Use local LLM (Ollama) to convert arbitrary questions → SQL.

**Requirements**:
- Ollama installed (https://ollama.ai/)
- Model pulled: `ollama pull llama3.1:8b`

**Prompt Structure**:
```
You are a SQL query generator for Formula 1 race data. Your ONLY job is to 
convert natural language questions into SQL queries.

Database Schema:
[Full schema with tables, columns, sample rows]

Rules:
1. You MUST generate ONLY a SQL SELECT query. Do not include any explanation.
2. The query must be valid SQLite syntax.
3. Use exact column names and table names from the schema above.
4. For event names, use the exact format from the database 
   (e.g., "Monaco Grand Prix" not just "Monaco").
5. For years, use integer comparison (e.g., event_year = 2023).

Example queries:
- "Who had the fastest lap in Monaco 2023?" → 
  SELECT Driver, MIN(LapTime) FROM laps WHERE ...

Now generate SQL for this question:
[User's question]

SQL:
```

**How It Works**:
1. Load database schema
2. Build prompt with schema + question
3. Call Ollama via `subprocess`
4. Extract SQL from response (handles markdown code blocks)
5. Return clean SQL string

**Example**:
```python
from backend.chatbot.nl_to_sql import translate_question_to_sql

sql = translate_question_to_sql(
    "Show me all races where Verstappen finished in the top 3 in 2023"
)
# Returns: SELECT * FROM results WHERE Driver = 'Max Verstappen' ...
```

---

### 5. Main Chatbot (`chatbot.py`)

**Purpose**: Orchestrate the entire flow.

**Class: `F1Chatbot`**

**Main Method: `ask(question, use_llm=True)`**

**Flow**:
1. Try rule-based handler first
2. If fails, try LLM (if `use_llm=True`)
3. Validate generated SQL
4. Execute SQL against SQLite
5. Format response (answer + rows + explanation)

**Response Format**:
```python
{
    "answer": "Max Verstappen had the fastest lap with a time of 1:15.650.",
    "rows": [
        {
            "Driver": "Max Verstappen",
            "TeamName": "Red Bull Racing",
            "FastestLapTime": "0 days 01:15.650000",
            "event_year": 2023,
            "event_name": "Monaco Grand Prix"
        }
    ],
    "explanation": "I executed a SQL query against the F1 database...",
    "sql": "SELECT Driver, MIN(LapTime) as FastestLapTime FROM laps WHERE ...",
    "method": "rule_based",  # or "llm"
    "row_count": 1,
    "error": None
}
```

---

## Database Schema

The chatbot understands this schema (generated from actual data):

### `results` Table
```sql
CREATE TABLE results (
    event_year INTEGER,
    event_name TEXT,
    session_type TEXT,  -- 'Q' or 'R'
    Driver TEXT,
    TeamName TEXT,
    DriverNumber INTEGER,
    Position INTEGER,
    GridPosition INTEGER,
    Q1 TEXT,  -- Qualifying times
    Q2 TEXT,
    Q3 TEXT,
    FastestLap INTEGER,
    FastestLapTime TEXT,
    Points REAL,
    Status TEXT
);
```

### `laps` Table
```sql
CREATE TABLE laps (
    event_year INTEGER,
    event_name TEXT,
    session_type TEXT,
    Driver TEXT,
    TeamName TEXT,
    DriverNumber INTEGER,
    LapNumber INTEGER,
    LapTime TEXT,  -- Stored as string
    Position INTEGER,
    Sector1Time TEXT,
    Sector2Time TEXT,
    Sector3Time TEXT,
    Compound TEXT,
    TyreLife INTEGER,
    IsPersonalBest INTEGER,
    IsFastest INTEGER
);
```

### `weather` Table
```sql
CREATE TABLE weather (
    event_year INTEGER,
    event_name TEXT,
    session_type TEXT,
    AirTemp REAL,
    TrackTemp REAL,
    Humidity REAL,
    Rainfall REAL,
    WindSpeed REAL,
    WindDirection REAL
);
```

---

## Example Queries

### Rule-based Examples

**Question**: "Who had the fastest lap in Monaco 2023?"

**Generated SQL**:
```sql
SELECT 
    Driver,
    TeamName,
    MIN(LapTime) as FastestLapTime,
    event_year,
    event_name
FROM laps
WHERE event_year = 2023
    AND event_name = 'Monaco Grand Prix'
    AND session_type = 'R'
    AND LapTime IS NOT NULL
    AND LapTime != 'NaT'
    AND LapTime != ''
GROUP BY Driver, TeamName, event_year, event_name
ORDER BY FastestLapTime ASC
LIMIT 1
```

**Response**:
```json
{
  "answer": "Max Verstappen had the fastest lap with a time of 1:15.650.",
  "method": "rule_based",
  "rows": [{"Driver": "Max Verstappen", ...}]
}
```

### LLM-based Examples

**Question**: "Show me all races where Verstappen finished in the top 3 in 2023"

**Generated SQL** (by LLM):
```sql
SELECT 
    Driver,
    TeamName,
    Position,
    event_year,
    event_name
FROM results
WHERE Driver LIKE '%Verstappen%'
    AND event_year = 2023
    AND session_type = 'R'
    AND Position <= 3
ORDER BY event_name
```

**Response**:
```json
{
  "answer": "I found 15 result(s). See the detailed rows below.",
  "method": "llm",
  "rows": [...]
}
```

---

## Safety Mechanisms

### 1. SQL Validation

All SQL is validated before execution:

```python
# Allowed
SELECT * FROM results WHERE event_year = 2023

# Blocked
DROP TABLE results  # ❌ Dangerous keyword
INSERT INTO ...     # ❌ Write operation
DELETE FROM ...     # ❌ Write operation
```

### 2. Read-only Database

- SQLite connection is read-only by default
- No write operations possible
- Database file is separate from source Parquet files

### 3. LLM Constraint

- LLM is instructed to ONLY generate SQL
- No answer generation from LLM knowledge
- All answers come from query results

### 4. Error Handling

- If SQL generation fails → returns error message
- If SQL execution fails → returns error with SQL
- If no data found → explains why (year/race not in dataset)

---

## Setup Instructions

### Step 1: Build Database

```bash
python -m backend.chatbot.db_loader
```

This creates `data/f1_chatbot.db` from Parquet files.

### Step 2: Install Ollama (Optional)

```bash
# Install from https://ollama.ai/
# Pull model:
ollama pull llama3.1:8b
```

**Note**: Rule-based queries work without Ollama. LLM is only needed for arbitrary questions.

### Step 3: Test

```python
from backend.chatbot import ask_question

response = ask_question("Who had the fastest lap in Monaco 2023?")
print(response["answer"])
```

### Step 4: Use in API

The chatbot is integrated into the FastAPI server:

```bash
python run_api.py
# POST /chat?question=Who%20won%20Bahrain%202024
```

---

## Design Decisions

### Why Hybrid Approach?

- **Rule-based**: Fast, accurate, no dependencies
- **LLM fallback**: Handles arbitrary questions
- **Best of both worlds**: Speed for common queries, flexibility for complex ones

### Why SQLite?

- Fast local queries
- Standard SQL interface
- Easy to inspect/debug
- Parquet → SQLite conversion is straightforward

### Why Ollama (not OpenAI)?

- **Free**: No API costs
- **Local**: Data stays on your machine
- **Private**: No data sent to cloud
- **Offline**: Works without internet

### Why Read-only SQL?

- Safety: Prevents accidental data modification
- Intent: Chatbot is for querying, not modifying
- Validation: Easy to enforce with keyword checking

---

## Limitations

1. **Event name matching**: Rule-based handlers require exact race names. LLM is more flexible.

2. **Column availability**: Some columns may not exist in older data (e.g., `Points` might be missing).

3. **LLM accuracy**: The LLM may generate incorrect SQL for very complex queries. Always check `response["sql"]`.

4. **Performance**: LLM queries take 1-5 seconds (vs < 50ms for rule-based).

---

## Future Enhancements

- [ ] More rule-based handlers (podium, DNFs, championship standings)
- [ ] Query caching for common questions
- [ ] Better error messages with query suggestions
- [ ] Support for date ranges ("all races in 2023-2024")
- [ ] Comparison queries ("compare Verstappen vs Hamilton")
- [ ] Visualization support (charts from query results)

---

## Testing

```python
# Test rule-based
response = ask_question("Who won Bahrain 2024?", use_llm=False)

# Test LLM
response = ask_question("Show me all races where Hamilton got pole in 2023", use_llm=True)

# Check SQL
print(response["sql"])

# Verify answer
print(response["answer"])
print(response["rows"])
```

---

## Conclusion

This chatbot system demonstrates:
- ✅ **Free & local**: No paid APIs, all data stays local
- ✅ **Safe**: Read-only SQL, validated queries
- ✅ **Accurate**: Answers from dataset, not LLM knowledge
- ✅ **Fast**: Rule-based handlers for common queries
- ✅ **Flexible**: LLM fallback for arbitrary questions

The system acts as a **data analyst**, not a trivia bot, always sourcing answers from the actual dataset.

