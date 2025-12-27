# Installing Ollama (Optional)

Ollama is only needed if you want to ask **arbitrary questions** that aren't covered by the rule-based handlers.

**Rule-based queries work without Ollama!** These include:
- "Who had the fastest lap in Monaco 2023?"
- "Who won the Bahrain Grand Prix in 2024?"
- "Who got pole position in Monaco 2023?"

You only need Ollama for complex questions like:
- "Show me all races where Verstappen finished in the top 3 in 2023"
- "Which driver had the most fastest laps in 2024?"

---

## macOS Installation

### Option 1: Using Homebrew (Recommended)

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Ollama
brew install ollama
```

### Option 2: Direct Download

1. Visit https://ollama.ai/
2. Download the macOS installer
3. Run the installer
4. Ollama will be installed to `/usr/local/bin/ollama`

### Option 3: Using curl (Alternative)

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

---

## Verify Installation

```bash
# Check if Ollama is installed
which ollama

# Check version
ollama --version

# List available models (will be empty until you pull one)
ollama list
```

---

## Pull the Model

Once Ollama is installed, pull the Llama 3.1 model:

```bash
ollama pull llama3.1:8b
```

This will download ~4.7GB. The first time you use it, it may take a minute to start.

**Alternative models** (if you prefer):
```bash
ollama pull llama3:8b      # Llama 3 (slightly older)
ollama pull mistral        # Mistral 7B (smaller, faster)
ollama pull codellama:7b   # Code-focused model (good for SQL)
```

---

## Test Ollama

```bash
# Test with a simple prompt
ollama run llama3.1:8b "What is SQL?"

# Or test the chatbot with LLM enabled
python -c "from backend.chatbot import ask_question; print(ask_question('Show me all races in 2023', use_llm=True))"
```

---

## Troubleshooting

### "command not found" after installation

Add Ollama to your PATH. It's usually installed to `/usr/local/bin/ollama`:

```bash
# Check if it exists
ls -la /usr/local/bin/ollama

# If it exists, add to PATH (add to ~/.zshrc or ~/.bash_profile)
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### "Connection refused" error

Ollama runs as a local server. Make sure it's running:

```bash
# Start Ollama (usually auto-starts, but can manually start)
ollama serve

# In another terminal, test it
ollama list
```

### Model download fails

Check your internet connection and disk space. The model is ~4.7GB.

```bash
# Check disk space
df -h

# Try pulling again
ollama pull llama3.1:8b
```

---

## Using the Chatbot Without Ollama

You can still use the chatbot with rule-based handlers only:

```python
from backend.chatbot import ask_question

# This works without Ollama
response = ask_question("Who had the fastest lap in Monaco 2023?", use_llm=False)
print(response["answer"])

# This will fail gracefully if Ollama is not available
response = ask_question("Show me all races in 2023", use_llm=True)
# Will return: "Ollama not available" message
```


