import ask_question

# Rule-based (fast, no LLM needed)
response = ask_question("Who had the fastest lap in Monaco 2023?", use_llm=False)
print(response["answer"])

# With LLM fallback (for arbitrary questions)
response = ask_question("Show me all races where Verstappen finished in the top 3 in 2023")
print(response["answer"])
print(response["sql"])  # See the generated SQL