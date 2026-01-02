"""
Main chatbot orchestrator: Combines rule-based and LLM-based query generation.
"""
import sqlite3
import pandas as pd
import re
from typing import Dict, Any, Optional
from pathlib import Path

from .db_loader import DB_PATH
from .sql_validator import validate_sql, sanitize_sql
# Rule-based handlers removed - always use LLM for better interpretation
from .nl_to_sql import translate_question_to_sql, check_llm_available


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
        
        self.llm_available = check_llm_available()
    
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
            # Parse error to provide helpful guidance
            error_lower = error.lower()
            guidance = ""
            
            if "ambiguous column" in error_lower or "column name" in error_lower:
                guidance = "\n\n💡 **Tip**: The query has ambiguous column names. Try rephrasing your question more specifically, like:\n" \
                          "• \"Who won the 2023 championship?\" (instead of complex joins)\n" \
                          "• \"Which driver had the most points in 2024?\"\n" \
                          "• \"Show me Verstappen's race results in 2023\""
            elif "no such table" in error_lower or "table" in error_lower and "not exist" in error_lower:
                guidance = "\n\n💡 **Tip**: The query references a table that doesn't exist. Available tables are: results, laps, weather.\n" \
                          "Try rephrasing your question using these tables."
            elif "syntax error" in error_lower or "near" in error_lower:
                guidance = "\n\n💡 **Tip**: There's a syntax issue with the generated query. Try:\n" \
                          "• Being more specific about what you want\n" \
                          "• Including a year in your question (e.g., \"in 2023\")\n" \
                          "• Using simpler phrasing"
            elif "invalid" in error_lower:
                guidance = "\n\n💡 **Tip**: The query couldn't be executed. Try rephrasing your question:\n" \
                          "• Be more specific (include year, driver name, race name)\n" \
                          "• Use simpler language\n" \
                          "• Ask about one thing at a time\n\n" \
                          "**Examples that work well:**\n" \
                          "• \"Who won the 2023 championship?\"\n" \
                          "• \"Who had the fastest lap in Monaco 2023?\"\n" \
                          "• \"How many podiums did Verstappen get in 2024?\""
            else:
                guidance = "\n\n💡 **Tip**: Try rephrasing your question with:\n" \
                          "• A specific year (2018-2025)\n" \
                          "• Clear, simple language\n" \
                          "• One question at a time\n\n" \
                          "**Example questions:**\n" \
                          "• \"Who won most races in 2023?\"\n" \
                          "• \"Who was the best driver in 2022?\"\n" \
                          "• \"Show me all races in 2024\""
            
            # Create a friendly message that doesn't expose technical details
            friendly_message = "I had trouble understanding your question. Let me help you rephrase it."
            
            return {
                "answer": f"{friendly_message}{guidance}",
                "rows": [],
                "explanation": f"The query couldn't be executed. This usually means the question needs to be rephrased for better clarity.",
                "sql": sql,
                "error": error,
            }
        
        if df.empty:
            # Provide helpful guidance based on question type
            question_lower = question.lower()
            guidance = ""
            
            if any(word in question_lower for word in ["2026", "2027", "future", "next", "upcoming"]):
                guidance = "\n\nNote: I only have data from 2018-2025. Questions about 2026 or future seasons can't be answered yet."
            elif any(word in question_lower for word in ["race", "grand prix", "monaco", "bahrain"]):
                guidance = "\n\nTip: Make sure you're using the full race name (e.g., 'Monaco Grand Prix') and a valid year (2018-2025)."
            elif any(word in question_lower for word in ["driver", "team"]):
                guidance = "\n\nTip: Try using the driver's full name or check if they raced in the specified year."
            
            return {
                "answer": f"I couldn't find any data matching your question in the dataset.{guidance}\n\n"
                          "Available data: 2018-2025 seasons\n"
                          "Try rephrasing with a specific year and race name.",
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
            return "I couldn't find any data matching your question. Try rephrasing with a specific year (2018-2025) and clearer criteria."
        
        # Simple heuristics based on question type
        question_lower = question.lower()
        
        # Aggregate answers (counts, sums)
        if len(df.columns) == 1 and len(df) == 1:
            val = df.iloc[0, 0]
            if "how many" in question_lower:
                return f"The count is {val}."
            return f"The answer is {val}."

        # Single row with specific columns
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
                # Check if it was a count of wins
                wins = row.get("Wins", row.get("wins", None))
                if wins is not None:
                    return f"{driver} won the most races with {wins} victories."
                return f"{driver}{team_str} won the race."
            
            # Handle specific positions (2nd, 3rd, 5th, etc.) in races
            elif any(ord in question_lower for ord in ["2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "second", "third", "finished", "got", "placed"]):
                driver = row.get("Driver", "Unknown")
                team = row.get("TeamName", "")
                position = row.get("Position", "Unknown")
                team_str = f" ({team})" if team else ""
                
                # Get position suffix
                pos_num = int(position) if str(position).isdigit() else None
                if pos_num:
                    if pos_num == 1:
                        suffix = "st"
                    elif pos_num == 2:
                        suffix = "nd"
                    elif pos_num == 3:
                        suffix = "rd"
                    else:
                        suffix = "th"
                    return f"{driver}{team_str} finished {position}{suffix}."
                else:
                    return f"{driver}{team_str} finished in position {position}."
            
            elif "pole" in question_lower:
                driver = row.get("Driver", "Unknown")
                team = row.get("TeamName", "")
                team_str = f" ({team})" if team else ""
                poles = row.get("Poles", row.get("poles", None))
                if poles is not None:
                    return f"{driver} had the most pole positions ({poles})."
                return f"{driver}{team_str} started from pole position."
            
            elif "best" in question_lower or ("points" in question_lower and "most" in question_lower):
                driver = row.get("Driver", row.get("TeamName", "Unknown"))
                points = row.get("TotalPoints", row.get("Points", "Unknown"))
                return f"{driver} was the top performer with {points} points."
            
            elif "worst" in question_lower or "fewest" in question_lower or "least" in question_lower:
                driver = row.get("Driver", row.get("TeamName", "Unknown"))
                points = row.get("TotalPoints", row.get("Points", "Unknown"))
                return f"{driver} scored the fewest points with {points} points."
            
            elif "championship" in question_lower or "champion" in question_lower:
                driver = row.get("Driver", "Unknown")
                team = row.get("TeamName", "")
                points = row.get("TotalPoints", "Unknown")
                team_str = f" ({team})" if team else ""
                
                # Detect position in championship (2nd, 3rd, etc.)
                if "2nd" in question_lower or "second" in question_lower or "2" in question_lower:
                    return f"{driver}{team_str} finished 2nd in the championship with {points} points."
                elif "3rd" in question_lower or "third" in question_lower or "3" in question_lower:
                    return f"{driver}{team_str} finished 3rd in the championship with {points} points."
                elif any(ord in question_lower for ord in ["4th", "5th", "6th", "7th", "8th", "9th", "10th"]):
                    # Extract the number
                    import re
                    pos_match = re.search(r'(\d+)(?:st|nd|rd|th)', question_lower)
                    if pos_match:
                        pos = pos_match.group(1)
                        return f"{driver}{team_str} finished {pos}th in the championship with {points} points."
                
                return f"{driver}{team_str} won the championship with {points} points."
            
            else:
                # Generic single row answer
                driver = row.get("Driver", row.get("TeamName", ""))
                if driver:
                    return f"The result is {driver}."
                return str(row.iloc[0])
        
        # Multiple rows - summarize
        else:
            count = len(df)
            # Check if we have a Driver or TeamName column to list a few
            if "Driver" in df.columns:
                top_items = df["Driver"].head(3).tolist()
                items_str = ", ".join(top_items)
                if count > 3:
                    return f"I found {count} results. The top ones include: {items_str}."
                return f"I found {count} results: {items_str}."
            
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
    
    def _validate_question(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Minimal validation - let the LLM handle interpretation.
        Only blocks clearly unanswerable questions.
        
        Args:
            question: User's question
            
        Returns:
            Error response dict if question is problematic, None otherwise
        """
        question_lower = question.lower()
        
        # Only block questions about future data we don't have
        if any(word in question_lower for word in ["2026", "2027", "2028", "2029", "2030", "future", "next season"]):
            return {
                "answer": "I only have data from 2018-2025. I can't answer questions about 2026 or future seasons yet.\n\n"
                          "Try asking about historical data:\n"
                          "• \"Who won the 2025 championship?\"\n"
                          "• \"What happened in Monaco 2024?\"\n"
                          "• \"Compare drivers in 2023\"",
                "rows": [],
                "explanation": "Question asks about data outside available range (2018-2025).",
                "sql": None,
                "method": "validation",
                "error": "Future data requested",
            }
        
        # Let everything else through - let the LLM interpret it
        return None
    
    def ask(self, question: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        Main method: Answer a natural language question.
        
        Args:
            question: User's question
            use_llm: Whether to use LLM fallback if rule-based fails
            
        Returns:
            Dict with answer, rows, explanation, sql, etc.
        """
        # Validate question first
        validation_error = self._validate_question(question)
        if validation_error:
            return validation_error
        
        sql = None
        error = None
        method = "llm"
        
        # Always use LLM for all questions (no rule-based handlers)
        if use_llm:
            if not self.llm_available:
                import os
                if os.getenv("GROQ_API_KEY"):
                    error_msg = "Groq API key is set but API call failed. Check your API key at https://console.groq.com/"
                else:
                    error_msg = "LLM is not available. Options:\n" \
                               "1. Install Ollama: https://ollama.ai/ (then: ollama pull deepseek-coder:6.7b)\n" \
                               "2. OR set GROQ_API_KEY env var for cloud LLM (free tier available)"
                return {
                    "answer": f"I couldn't understand your question and LLM is not available.\n\n{error_msg}",
                    "rows": [],
                    "explanation": "LLM handler not available.",
                    "sql": None,
                    "method": "none",
                    "error": "LLM not available",
                }
            
            try:
                sql = translate_question_to_sql(question)
                method = "llm"
            except Exception as e:
                error = str(e)
                sql = None
        
        if not sql:
            return {
                "answer": "I couldn't understand your question. Here are some examples of what I can answer:\n\n"
                         "✅ \"Who won most races in 2023?\"\n"
                         "✅ \"Who had the fastest lap in Monaco 2023?\"\n"
                         "✅ \"Who scored the most points in 2022?\"\n"
                         "✅ \"Who got pole position in Bahrain 2024?\"\n"
                         "✅ \"How many podiums did Verstappen get in 2023?\"\n\n"
                         "Try rephrasing your question with:\n"
                         "• A specific year (e.g., \"in 2023\")\n"
                         "• Clear criteria (e.g., \"most points\", \"fastest lap\")\n"
                         "• Objective facts rather than opinions",
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

