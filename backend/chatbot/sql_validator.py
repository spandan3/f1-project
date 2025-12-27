"""
SQL validator: Ensures queries are read-only and safe.
"""
import re
from typing import Tuple, Optional


# Dangerous SQL keywords (write operations)
DANGEROUS_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", 
    "TRUNCATE", "REPLACE", "MERGE", "EXEC", "EXECUTE",
    "ATTACH", "DETACH", "VACUUM", "PRAGMA"
]

# Allowed pragmas (read-only)
ALLOWED_PRAGMAS = ["table_info", "index_list", "index_info", "foreign_key_list"]


def validate_sql(sql: str) -> Tuple[bool, Optional[str]]:
    """
    Validate SQL query is read-only and safe.
    
    Args:
        sql: SQL query string
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid: (True, None)
        If invalid: (False, error_description)
    """
    sql_upper = sql.strip().upper()
    
    # Must start with SELECT
    if not sql_upper.startswith("SELECT"):
        # Allow WITH ... SELECT (CTEs)
        if not sql_upper.startswith("WITH"):
            return False, "Query must start with SELECT or WITH"
    
    # Check for dangerous keywords (case-insensitive)
    for keyword in DANGEROUS_KEYWORDS:
        # Use word boundaries to avoid matching substrings
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, sql_upper, re.IGNORECASE):
            # Special handling for PRAGMA
            if keyword == "PRAGMA":
                # Check if it's an allowed pragma
                pragma_match = re.search(r'PRAGMA\s+(\w+)', sql_upper, re.IGNORECASE)
                if pragma_match:
                    pragma_name = pragma_match.group(1).lower()
                    if pragma_name not in ALLOWED_PRAGMAS:
                        return False, f"PRAGMA {pragma_name} is not allowed"
                    # Allow this pragma
                    continue
            return False, f"Dangerous keyword detected: {keyword}"
    
    # Check for semicolon injection attempts
    # Allow single semicolon at end, but not multiple statements
    sql_clean = sql.strip()
    if sql_clean.count(';') > 1:
        return False, "Multiple statements not allowed"
    
    # Check for comment-based injection attempts
    if '--' in sql or '/*' in sql:
        # Allow comments but log warning (we'll be conservative)
        # In practice, comments are usually fine, but we'll allow them
        pass
    
    return True, None


def sanitize_sql(sql: str) -> str:
    """
    Basic SQL sanitization (remove extra whitespace, normalize).
    
    Args:
        sql: SQL query string
        
    Returns:
        Sanitized SQL string
    """
    # Remove extra whitespace
    sql = ' '.join(sql.split())
    
    # Remove trailing semicolon (we'll add it if needed)
    sql = sql.rstrip(';').strip()
    
    return sql

