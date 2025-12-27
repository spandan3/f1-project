"""
Main chatbot orchestrator: Combines rule-based and LLM-based query generation.
"""
import sqlite3
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path

from .db_loader import DB_PATH
from .sql_validator import validate_sql, sanitize_sql
from .rule_handlers import get_rule_based_sql
from .nl_to_sql import translate_question_to_sql, check_ollama_available


class F1Chatbot:
    """
    F1 data chatbot that answers questions using local SQLite database.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize chatbot.
        
        Args:
            db_path: Path to SQLite database (default: from db_loader)
        """
        self.db_path = db_path or DB_PATH
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found: {self.db_path}. "
                "Run: python -m backend.chatbot.db_loader to create it."
            )
        
        self.ollama_available = check_ollama_available()
    
    def execute_query(self, sql: str) -> tuple[pd.DataFrame, Optional[str]]:
        """
        Execute SQL query safely.
        
        Args:
            sql: SQL query string
            
        Returns:
            Tuple of (DataFrame with results, error_message)
        """
        # Validate SQL
        is_valid, error = validate_sql(sql)
        if not is_valid:
            return pd.DataFrame(), error
        
        # Sanitize
        sql = sanitize_sql(sql)
        
        # Validate tables exist
        table_error = self._validate_tables_in_sql(sql)
        if table_error:
            return pd.DataFrame(), table_error
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            df = pd.read_sql_query(sql, conn)
            conn.close()
            return df, None
        except Exception as e:
            return pd.DataFrame(), str(e)
    
    def _validate_tables_in_sql(self, sql: str) -> Optional[str]:
        """
        Check if all tables referenced in SQL actually exist.
        
        Args:
            sql: SQL query string
            
        Returns:
            Error message if invalid, None if valid
        """
        import re
        
        # Get list of actual tables
        conn = sqlite3.connect(str(self.db_path))
        actual_tables = {
            row[0] for row in 
            conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        conn.close()
        
        # Extract table names from SQL (simple regex, handles FROM and JOIN)
        # Look for FROM table_name and JOIN table_name patterns
        from_pattern = r'\bFROM\s+(\w+)'
        join_pattern = r'\bJOIN\s+(\w+)'
        
        referenced_tables = set()
        for match in re.finditer(from_pattern, sql, re.IGNORECASE):
            referenced_tables.add(match.group(1).lower())
        for match in re.finditer(join_pattern, sql, re.IGNORECASE):
            referenced_tables.add(match.group(1).lower())
        
        # Check each referenced table
        actual_tables_lower = {t.lower() for t in actual_tables}
        for table in referenced_tables:
            if table not in actual_tables_lower:
                return f"Table '{table}' does not exist. Available tables: {', '.join(sorted(actual_tables))}"
        
        return None
    
    def format_response(self, question: str, df: pd.DataFrame, sql: str, 
                       error: Optional[str] = None) -> Dict[str, Any]:
        """
        Format chatbot response.
        
        Args:
            question: Original user question
            df: Query results DataFrame
            sql: SQL query used
            error: Error message if any
            
        Returns:
            Dict with answer, rows, explanation, and metadata
        """
        if error:
            return {
                "answer": f"I encountered an error: {error}",
                "rows": [],
                "explanation": f"Could not execute the generated SQL query.",
                "sql": sql,
                "error": error,
            }
        
        if df.empty:
            return {
                "answer": "I couldn't find any data matching your question in the dataset.",
                "rows": [],
                "explanation": "The query executed successfully but returned no results. "
                             "This might mean the race/year doesn't exist in the dataset, "
                             "or the query criteria didn't match any records.",
                "sql": sql,
                "error": None,
            }
        
        # Generate natural language answer
        answer = self._generate_answer(df, question)
        
        # Convert DataFrame to list of dicts for JSON response
        rows = df.to_dict(orient="records")
        
        # Generate explanation
        explanation = self._generate_explanation(question, df, sql)
        
        return {
            "answer": answer,
            "rows": rows,
            "explanation": explanation,
            "sql": sql,
            "row_count": len(df),
            "error": None,
        }
    
    def _generate_answer(self, df: pd.DataFrame, question: str) -> str:
        """
        Generate natural language answer from query results.
        
        Args:
            df: Query results
            question: Original question
            
        Returns:
            Natural language answer string
        """
        if df.empty:
            return "I couldn't find any data matching your question."
        
        # Simple heuristics based on question type
        question_lower = question.lower()
        
        # Single value answers (fastest lap, winner, etc.)
        if len(df) == 1:
            row = df.iloc[0]
            
            if "fastest" in question_lower or "best lap" in question_lower:
                driver = row.get("Driver", "Unknown")
                lap_time = row.get("FastestLapTime", row.get("LapTime", "Unknown"))
                return f"{driver} had the fastest lap with a time of {lap_time}."
            
            elif "won" in question_lower or "winner" in question_lower:
                driver = row.get("Driver", "Unknown")
                team = row.get("TeamName", "")
                team_str = f" for {team}" if team else ""
                return f"{driver}{team_str} won the race."
            
            elif "pole" in question_lower:
                driver = row.get("Driver", "Unknown")
                team = row.get("TeamName", "")
                team_str = f" ({team})" if team else ""
                return f"{driver}{team_str} started from pole position."
            
            else:
                # Generic single row answer
                driver = row.get("Driver", "")
                if driver:
                    return f"{driver}"
                return str(row.iloc[0])
        
        # Aggregate answers (most points, etc.)
        elif "most" in question_lower or "highest" in question_lower:
            row = df.iloc[0]  # Should be sorted
            team = row.get("TeamName", "")
            driver = row.get("Driver", "")
            points = row.get("TotalPoints", row.get("Points", ""))
            
            if team:
                return f"{team} scored the most points ({points})."
            elif driver:
                return f"{driver} scored the most points ({points})."
            else:
                return str(row.iloc[0])
        
        # List answers
        else:
            count = len(df)
            return f"I found {count} result(s). See the detailed rows below."
    
    def _generate_explanation(self, question: str, df: pd.DataFrame, sql: str) -> str:
        """
        Generate explanation of how the answer was derived.
        
        Args:
            question: Original question
            df: Query results
            sql: SQL query used
            
        Returns:
            Explanation string
        """
        row_count = len(df)
        
        explanation = f"I executed a SQL query against the F1 database and found {row_count} matching record(s). "
        
        # Add context based on query
        if "fastest" in question.lower() or "best lap" in question.lower():
            explanation += "The query searched the laps table for the minimum lap time in the race session."
        elif "won" in question.lower() or "winner" in question.lower():
            explanation += "The query searched for the driver who finished in position 1 in the race results."
        elif "pole" in question.lower():
            explanation += "The query searched for the driver who qualified in position 1."
        else:
            explanation += "The query retrieved the relevant records from the database."
        
        return explanation
    
    def ask(self, question: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        Main method: Answer a natural language question.
        
        Args:
            question: User's question
            use_llm: Whether to use LLM fallback if rule-based fails
            
        Returns:
            Dict with answer, rows, explanation, sql, etc.
        """
        sql = None
        error = None
        
        # Try rule-based first (fast and accurate)
        sql = get_rule_based_sql(question)
        method = "rule_based"
        
        # Fallback to LLM if rule-based fails
        if not sql and use_llm:
            if not self.ollama_available:
                return {
                    "answer": "I couldn't understand your question and Ollama (LLM) is not available. "
                             "Please install Ollama and pull a model: ollama pull llama3.1:8b",
                    "rows": [],
                    "explanation": "Rule-based handler didn't match the question.",
                    "sql": None,
                    "method": "none",
                    "error": "Ollama not available",
                }
            
            try:
                sql = translate_question_to_sql(question)
                method = "llm"
            except Exception as e:
                error = str(e)
                sql = None
        
        if not sql:
            return {
                "answer": "I couldn't generate a SQL query for your question. "
                         "Please try rephrasing it or ask about fastest laps, race winners, or pole positions.",
                "rows": [],
                "explanation": "Neither rule-based handler nor LLM could generate a valid query.",
                "sql": None,
                "method": "none",
                "error": error or "Query generation failed",
            }
        
        # Execute query
        df, exec_error = self.execute_query(sql)
        
        if exec_error:
            error = exec_error
        
        # Format response
        response = self.format_response(question, df, sql, error)
        response["method"] = method
        
        return response


# Convenience function
def ask_question(question: str, use_llm: bool = True) -> Dict[str, Any]:
    """
    Quick function to ask the chatbot a question.
    
    Args:
        question: User's question
        use_llm: Whether to use LLM fallback
        
    Returns:
        Response dict
    """
    chatbot = F1Chatbot()
    return chatbot.ask(question, use_llm=use_llm)


if __name__ == "__main__":
    # Example usage
    chatbot = F1Chatbot()
    
    test_questions = [
        "Who had the fastest lap in Monaco 2023?",
        "Who won the Bahrain Grand Prix in 2024?",
        "Who got pole position in Monaco 2023?",
    ]
    
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print('='*60)
        response = chatbot.ask(q)
        print(f"\nAnswer: {response['answer']}")
        print(f"\nMethod: {response['method']}")
        print(f"\nSQL: {response['sql']}")
        if response['rows']:
            print(f"\nRows: {len(response['rows'])} results")
            print(response['rows'])

