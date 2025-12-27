"""
Rule-based query handlers for common F1 questions.
These are fast, accurate, and don't require LLM calls.
"""
import re
from typing import Optional, Dict, Any
from pathlib import Path

from .db_loader import DB_PATH


# Common intent patterns
INTENT_PATTERNS = {
    "fastest_lap": [
        r"fastest\s+lap",
        r"who\s+had\s+the\s+fastest",
        r"fastest\s+laptime",
        r"best\s+lap",
    ],
    "race_winner": [
        r"who\s+won",
        r"winner",
        r"won\s+the\s+race",
        r"first\s+place",
        r"podium.*first",
    ],
    "pole_position": [
        r"pole\s+position",
        r"pole\s+sitter",
        r"who\s+got\s+pole",
        r"qualifying.*winner",
    ],
    "team_points": [
        r"team.*points",
        r"which\s+team.*most\s+points",
        r"constructor.*points",
        r"points.*team",
    ],
    "driver_points": [
        r"driver.*points",
        r"which\s+driver.*most\s+points",
    ],
}


def extract_year_and_race(question: str) -> tuple[Optional[int], Optional[str]]:
    """
    Extract year and race name from question.
    
    Args:
        question: Natural language question
        
    Returns:
        Tuple of (year, race_name) or (None, None) if not found
    """
    # Extract year (4 digits)
    year_match = re.search(r'\b(20\d{2})\b', question)
    year = int(year_match.group(1)) if year_match else None
    
    # Extract race name (common GPs)
    race_patterns = [
        r"Monaco\s+Grand\s+Prix",
        r"Bahrain\s+Grand\s+Prix",
        r"Monaco",
        r"Bahrain",
        r"Silverstone",
        r"British\s+Grand\s+Prix",
        r"Spa",
        r"Belgian\s+Grand\s+Prix",
        r"Monza",
        r"Italian\s+Grand\s+Prix",
        r"Singapore\s+Grand\s+Prix",
        r"Singapore",
    ]
    
    race_name = None
    for pattern in race_patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            race_name = match.group(0)
            # Normalize to full name
            if race_name.lower() == "monaco":
                race_name = "Monaco Grand Prix"
            elif race_name.lower() == "bahrain":
                race_name = "Bahrain Grand Prix"
            elif race_name.lower() == "silverstone" or race_name.lower() == "british":
                race_name = "British Grand Prix"
            elif race_name.lower() == "spa" or race_name.lower() == "belgian":
                race_name = "Belgian Grand Prix"
            elif race_name.lower() == "monza" or race_name.lower() == "italian":
                race_name = "Italian Grand Prix"
            elif race_name.lower() == "singapore":
                race_name = "Singapore Grand Prix"
            break
    
    return year, race_name


def handle_fastest_lap(question: str) -> Optional[str]:
    """
    Generate SQL for fastest lap queries.
    
    Example: "Who had the fastest lap in Monaco 2023?"
    
    Note: SQL uses string formatting but is validated before execution.
    The chatbot.execute_query() method will validate the SQL is read-only.
    """
    year, race_name = extract_year_and_race(question)
    
    if not year or not race_name:
        return None  # Cannot handle without year and race
    
    # Escape single quotes in race name (SQL injection prevention)
    race_name_escaped = race_name.replace("'", "''")
    
    # Query for fastest lap in race session
    sql = f"""
    SELECT 
        Driver,
        TeamName,
        MIN(LapTime) as FastestLapTime,
        event_year,
        event_name
    FROM laps
    WHERE event_year = {year}
        AND event_name = '{race_name_escaped}'
        AND session_type = 'R'
        AND LapTime IS NOT NULL
        AND LapTime != 'NaT'
        AND LapTime != ''
    GROUP BY Driver, TeamName, event_year, event_name
    ORDER BY FastestLapTime ASC
    LIMIT 1
    """
    
    return sql.strip()


def handle_race_winner(question: str) -> Optional[str]:
    """
    Generate SQL for race winner queries.
    
    Example: "Who won the Bahrain Grand Prix in 2024?"
    """
    year, race_name = extract_year_and_race(question)
    
    if not year or not race_name:
        return None
    
    race_name_escaped = race_name.replace("'", "''")
    
    sql = f"""
    SELECT 
        Driver,
        TeamName,
        Position as FinishPosition,
        event_year,
        event_name
    FROM results
    WHERE event_year = {year}
        AND event_name = '{race_name_escaped}'
        AND session_type = 'R'
        AND Position = 1
    LIMIT 1
    """
    
    return sql.strip()


def handle_pole_position(question: str) -> Optional[str]:
    """
    Generate SQL for pole position queries.
    
    Example: "Who got pole position in Monaco 2023?"
    """
    year, race_name = extract_year_and_race(question)
    
    if not year or not race_name:
        return None
    
    race_name_escaped = race_name.replace("'", "''")
    
    sql = f"""
    SELECT 
        Driver,
        TeamName,
        GridPosition,
        event_year,
        event_name
    FROM results
    WHERE event_year = {year}
        AND event_name = '{race_name_escaped}'
        AND session_type = 'Q'
        AND Position = 1
    LIMIT 1
    """
    
    return sql.strip()


def handle_team_points(question: str) -> Optional[str]:
    """
    Generate SQL for team points queries.
    
    Example: "Which team scored the most points in 2022?"
    """
    year_match = re.search(r'\b(20\d{2})\b', question)
    year = int(year_match.group(1)) if year_match else None
    
    if not year:
        return None
    
    # Check if Points column exists, otherwise we can't answer
    # For now, return a query that would work if Points exists
    sql = f"""
    SELECT 
        TeamName,
        SUM(CAST(Points AS REAL)) as TotalPoints
    FROM results
    WHERE event_year = {year}
        AND session_type = 'R'
        AND Points IS NOT NULL
    GROUP BY TeamName
    ORDER BY TotalPoints DESC
    LIMIT 1
    """
    
    return sql.strip()


def detect_intent(question: str) -> Optional[str]:
    """
    Detect query intent from question.
    
    Args:
        question: Natural language question
        
    Returns:
        Intent name (e.g., "fastest_lap") or None
    """
    question_lower = question.lower()
    
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, question_lower):
                return intent
    
    return None


def get_rule_based_sql(question: str) -> Optional[str]:
    """
    Try to generate SQL using rule-based handlers.
    
    Args:
        question: Natural language question
        
    Returns:
        SQL query string or None if no rule matches
    """
    intent = detect_intent(question)
    
    if intent == "fastest_lap":
        return handle_fastest_lap(question)
    elif intent == "race_winner":
        return handle_race_winner(question)
    elif intent == "pole_position":
        return handle_pole_position(question)
    elif intent == "team_points":
        return handle_team_points(question)
    
    return None

