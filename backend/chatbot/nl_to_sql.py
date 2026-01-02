"""
Natural Language to SQL translator using Ollama (local) or Groq (cloud).
"""
import json
import subprocess
import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

from .db_loader import get_db_schema

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_MODEL = os.getenv("CHABOT_MODEL", "deepseek-coder:6.7b")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # Default: fast model. Options: llama-3.3-70b-versatile (best), mixtral-8x7b-32768

# Use Groq if API key is set, otherwise use Ollama
USE_GROQ = GROQ_API_KEY is not None and GROQ_API_KEY != "your_groq_api_key_here"


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
    
    prompt = f"""You are an expert SQL query generator for a Formula 1 database (SQLite). 
Your task is to convert a user's question into a single, valid SQL query.

{schema}

---
CRITICAL KNOWLEDGE & RULES:
1. TABLES: Use ONLY {tables_list}. 
2. NO UNNECESSARY JOINS: 
   - Use 'results' table for: Winner, Position, Points, Grid, Driver, Team.
   - Use 'laps' table for: Lap times, Sector times, Tyre info.
   - DO NOT join 'laps' if the question only asks about positions or winners.
3. QUALIFY COLUMNS: Always use table aliases (e.g. r.Driver, l.LapTime) when joining to avoid 'ambiguous column' errors.
4. POSITIONS:
   - "Winner" or "P1" -> Position = 1
   - "Podium" -> Position <= 3
   - "2nd place" -> Position = 2
   - "Championship Position X" -> Use SUM(Points) GROUP BY Driver ORDER BY SUM(Points) DESC LIMIT 1 OFFSET (X-1)
5. SESSION TYPES:
   - 'R' = Race (ALWAYS use this for "who won", "points", "championship")
   - 'Q' = Qualifying (Use for "pole position")
6. STRINGS: Use LIKE with wildcards for names: Driver LIKE '%Verstappen%'.
7. SPEED: Fastest lap is MIN(LapTime) from 'laps' table.

EXAMPLES:
- "Who won the 2023 Monaco GP?"
  SELECT Driver FROM results WHERE event_year = 2023 AND event_name LIKE '%Monaco%' AND Position = 1 AND session_type = 'R'

- "Who got 2nd in 2024 championship?"
  SELECT Driver, SUM(CAST(Points AS REAL)) as TotalPoints FROM results WHERE event_year = 2024 AND session_type = 'R' AND Points IS NOT NULL GROUP BY Driver ORDER BY TotalPoints DESC LIMIT 1 OFFSET 1

- "Who finished 3rd in Monaco 2023?"
  SELECT Driver FROM results WHERE event_year = 2023 AND event_name LIKE '%Monaco%' AND Position = 3 AND session_type = 'R'

- "Who got 5th place in Bahrain 2024?"
  SELECT Driver FROM results WHERE event_year = 2024 AND event_name LIKE '%Bahrain%' AND Position = 5 AND session_type = 'R'

- "Who finished 19th in a race in 2025?"
  SELECT Driver FROM results WHERE event_year = 2025 AND Position = 19 AND session_type = 'R'

- "Alonso's podiums in 2023?"
  SELECT COUNT(*) FROM results WHERE Driver LIKE '%Alonso%' AND event_year = 2023 AND Position <= 3 AND session_type = 'R'

- "Fastest lap time in Silverstone 2024?"
  SELECT MIN(LapTime) FROM laps WHERE event_year = 2024 AND event_name LIKE '%British%' AND session_type = 'R'

- "Most points by a team in 2022?"
  SELECT TeamName, SUM(CAST(Points AS REAL)) as Total FROM results WHERE event_year = 2022 AND session_type = 'R' GROUP BY TeamName ORDER BY Total DESC LIMIT 1

- "Who won the 2025 championship?"
  SELECT Driver, SUM(CAST(Points AS REAL)) as TotalPoints FROM results WHERE event_year = 2025 AND session_type = 'R' AND Points IS NOT NULL GROUP BY Driver ORDER BY TotalPoints DESC LIMIT 1

- "Who was 3rd in 2023 championship?"
  SELECT Driver, SUM(CAST(Points AS REAL)) as TotalPoints FROM results WHERE event_year = 2023 AND session_type = 'R' AND Points IS NOT NULL GROUP BY Driver ORDER BY TotalPoints DESC LIMIT 1 OFFSET 2

---
Now, generate the SQL for this question: "{question}"
SQL:"""
    
    return prompt


def call_groq(prompt: str, model: str = GROQ_MODEL) -> str:
    """
    Call Groq Cloud API (faster, larger models).
    
    Args:
        prompt: Prompt to send
        model: Model name (default: llama-3.1-70b-versatile)
        
    Returns:
        LLM response string
    """
    try:
        from groq import Groq
        
        client = Groq(api_key=GROQ_API_KEY)
        
        # Try the requested model, fallback to available models if it fails
        models_to_try = [
            model,
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768"
        ]
        
        last_error = None
        for model_name in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert SQL query generator. Generate ONLY valid SQLite SELECT queries. No explanations, no markdown, just SQL."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,  # Low temperature for more consistent SQL
                    max_tokens=500,
                )
                break  # Success, exit loop
            except Exception as e:
                last_error = e
                if "model" in str(e).lower() and "decommissioned" in str(e).lower():
                    continue  # Try next model
                else:
                    raise  # Re-raise if it's not a model error
        
        if 'response' not in locals():
            raise RuntimeError(f"All models failed. Last error: {last_error}")
        
        return response.choices[0].message.content.strip()
    except ImportError:
        raise RuntimeError(
            "Groq package not installed. Install with: pip install groq\n"
            "Get API key from: https://console.groq.com/"
        )
    except Exception as e:
        raise RuntimeError(f"Groq API call failed: {str(e)}")


def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Call Ollama LLM API (local).
    
    Args:
        prompt: Prompt to send
        model: Model name (default: deepseek-coder:6.7b)
        
    Returns:
        LLM response string
    """
    try:
        # Call ollama via command line
        # Use UTF-8 encoding explicitly to avoid Windows cp1252 issues
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # Replace invalid UTF-8 bytes instead of failing
            timeout=30,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Ollama call failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError(
            "Ollama not found. Please install from https://ollama.ai/\n"
            "Then run: ollama pull deepseek-coder:6.7b"
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
    Translate natural language question to SQL using Groq or Ollama.
    
    Args:
        question: User's natural language question
        
    Returns:
        SQL query string
        
    Raises:
        RuntimeError: If LLM call fails or SQL extraction fails
    """
    schema = get_db_schema()
    prompt = generate_sql_prompt(question, schema)
    
    # Use Groq if available, otherwise Ollama
    if USE_GROQ:
        response = call_groq(prompt)
    else:
        response = call_ollama(prompt)
    
    # Extract SQL
    sql = extract_sql_from_response(response)
    
    if not sql or not sql.upper().startswith("SELECT"):
        raise RuntimeError(f"Failed to generate valid SQL. LLM response: {response[:200]}")
    
    return sql


def check_llm_available() -> bool:
    """
    Check if LLM (Groq or Ollama) is available.
    
    Returns:
        True if LLM is available, False otherwise
    """
    if USE_GROQ:
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            # Quick test call with fallback models
            models_to_try = [GROQ_MODEL, "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
            for model_name in models_to_try:
                try:
                    client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=1
                    )
                    return True
                except:
                    continue
            return False
        except:
            return False
    else:
        try:
            subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=True
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False


# Backward compatibility
def check_ollama_available() -> bool:
    """Backward compatibility alias."""
    return check_llm_available()

