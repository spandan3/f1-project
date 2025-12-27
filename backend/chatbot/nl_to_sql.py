"""
Natural Language to SQL translator using Ollama (local LLM).
"""
import json
import subprocess
from typing import Optional
from pathlib import Path

from .db_loader import get_db_schema


# Ollama model to use (must be installed locally)
OLLAMA_MODEL = "llama3.1:8b"  # or "llama3:8b", "mistral", etc.


def generate_sql_prompt(question: str, schema: str) -> str:
    """
    Generate prompt for LLM to translate NL to SQL.
    
    Args:
        question: User's natural language question
        schema: Database schema description
        
    Returns:
        Complete prompt string
    """
    # Extract table names from schema for emphasis
    import re
    table_matches = re.findall(r'## Table: (\w+)', schema)
    tables_list = ', '.join(table_matches) if table_matches else 'results, laps, weather'
    
    prompt = f"""You are a SQL query generator for Formula 1 race data. Your ONLY job is to convert natural language questions into SQL queries.

Database Schema:
{schema}

CRITICAL RULES:
1. Available tables are: {tables_list}. Use ONLY these table names.
2. DO NOT invent table names like "events", "races", "drivers", or "teams".
3. To list races/events, use: SELECT DISTINCT event_name FROM results WHERE event_year = 2023

Rules:
1. You MUST generate ONLY a SQL SELECT query. Do not include any explanation or markdown.
2. The query must be valid SQLite syntax.
3. Use exact column names and table names from the schema above.
4. For event names, use the exact format from the database (e.g., "Monaco Grand Prix" not just "Monaco").
5. For years, use integer comparison (e.g., event_year = 2023).
6. Return only the SQL query, nothing else.

Example queries:
- "Who had the fastest lap in Monaco 2023?" → SELECT Driver, MIN(LapTime) FROM laps WHERE event_year = 2023 AND event_name = 'Monaco Grand Prix' AND session_type = 'R' GROUP BY Driver ORDER BY MIN(LapTime) LIMIT 1
- "Who won the Bahrain Grand Prix in 2024?" → SELECT Driver FROM results WHERE event_year = 2024 AND event_name = 'Bahrain Grand Prix' AND session_type = 'R' AND Position = 1
- "Which team scored the most points in 2022?" → SELECT TeamName, SUM(Points) as TotalPoints FROM results WHERE event_year = 2022 AND session_type = 'R' GROUP BY TeamName ORDER BY TotalPoints DESC LIMIT 1
- "Show me all races in 2023" → SELECT DISTINCT event_name FROM results WHERE event_year = 2023
- "List all events in 2024" → SELECT DISTINCT event_name FROM results WHERE event_year = 2024

Now generate SQL for this question:
{question}

SQL:"""
    
    return prompt


def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Call Ollama LLM API.
    
    Args:
        prompt: Prompt to send
        model: Model name (default: llama3.1:8b)
        
    Returns:
        LLM response string
    """
    try:
        # Call ollama via command line
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=30,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Ollama call failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError(
            "Ollama not found. Please install from https://ollama.ai/\n"
            "Then run: ollama pull llama3.1:8b"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Ollama request timed out")


def extract_sql_from_response(response: str) -> str:
    """
    Extract SQL query from LLM response.
    May contain markdown code blocks or extra text.
    
    Args:
        response: Raw LLM response
        
    Returns:
        Cleaned SQL query
    """
    # Remove markdown code blocks
    sql = response.strip()
    
    # Remove ```sql ... ```
    if "```sql" in sql:
        sql = sql.split("```sql")[1].split("```")[0].strip()
    elif "```" in sql:
        sql = sql.split("```")[1].split("```")[0].strip()
    
    # Take first line that starts with SELECT
    lines = sql.split("\n")
    sql_lines = []
    in_query = False
    
    for line in lines:
        line_stripped = line.strip().upper()
        if line_stripped.startswith("SELECT") or line_stripped.startswith("WITH"):
            in_query = True
        
        if in_query:
            sql_lines.append(line)
            # Stop at semicolon or empty line after query starts
            if line.strip().endswith(";") or (line_stripped == "" and len(sql_lines) > 1):
                break
    
    sql = " ".join(sql_lines).strip()
    
    # Remove trailing semicolon
    sql = sql.rstrip(";").strip()
    
    return sql


def translate_question_to_sql(question: str) -> str:
    """
    Translate natural language question to SQL using Ollama.
    
    Args:
        question: User's natural language question
        
    Returns:
        SQL query string
        
    Raises:
        RuntimeError: If Ollama call fails or SQL extraction fails
    """
    schema = get_db_schema()
    prompt = generate_sql_prompt(question, schema)
    
    # Call Ollama
    response = call_ollama(prompt)
    
    # Extract SQL
    sql = extract_sql_from_response(response)
    
    if not sql or not sql.upper().startswith("SELECT"):
        raise RuntimeError(f"Failed to generate valid SQL. LLM response: {response[:200]}")
    
    return sql


def check_ollama_available() -> bool:
    """
    Check if Ollama is installed and model is available.
    
    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=5,
            check=True
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False

