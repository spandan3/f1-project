"""
Test script to verify Groq setup is working correctly.
Run this after setting up your .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

print("=" * 60)
print("🧪 Testing Groq Setup")
print("=" * 60)
print()

# Test 1: Check if .env file exists
print("1️⃣  Checking .env file...")
if env_path.exists():
    print("   ✅ .env file found")
else:
    print("   ❌ .env file not found!")
    print("   💡 Create it by copying env.example to .env")
    exit(1)

# Test 2: Check if API key is loaded
print("\n2️⃣  Checking GROQ_API_KEY...")
api_key = os.getenv("GROQ_API_KEY")
if api_key and api_key != "your_groq_api_key_here":
    print(f"   ✅ API key loaded (starts with: {api_key[:7]}...)")
else:
    print("   ❌ GROQ_API_KEY not set or still has placeholder value!")
    print("   💡 Make sure you've set GROQ_API_KEY in your .env file")
    exit(1)

# Test 3: Check if groq package is installed
print("\n3️⃣  Checking groq package...")
try:
    import groq
    print("   ✅ groq package installed")
except ImportError:
    print("   ❌ groq package not installed!")
    print("   💡 Run: pip install groq")
    exit(1)

# Test 4: Test Groq API connection
print("\n4️⃣  Testing Groq API connection...")
try:
    from groq import Groq
    
    client = Groq(api_key=api_key)
    
    # Simple test call - try multiple models
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    models_to_try = [model, "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
    
    response = None
    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": "Say 'Hello, Groq is working!' if you can read this."
                    }
                ],
                max_tokens=20,
            )
            print(f"   ✅ Using model: {model_name}")
            break
        except Exception as e:
            if "decommissioned" in str(e).lower():
                continue
            raise
    
    if not response:
        raise Exception("Could not find a working model")
    
    result = response.choices[0].message.content.strip()
    print(f"   ✅ Groq API is working!")
    print(f"   📝 Response: {result}")
    
except Exception as e:
    print(f"   ❌ Groq API test failed: {str(e)}")
    print("   💡 Check your API key and internet connection")
    exit(1)

# Test 5: Test chatbot integration
print("\n5️⃣  Testing chatbot integration...")
try:
    from backend.chatbot import ask_question
    
    # Simple test question
    test_question = "Who won the 2023 championship?"
    print(f"   📝 Test question: '{test_question}'")
    
    response = ask_question(test_question, use_llm=True)
    
    if response.get("error"):
        print(f"   ⚠️  Warning: {response.get('error')}")
    else:
        print(f"   ✅ Chatbot responded successfully!")
        print(f"   📝 Answer: {response.get('answer', 'No answer')[:100]}...")
        print(f"   🔧 Method: {response.get('method', 'unknown')}")
        
        if response.get("sql"):
            print(f"   📊 SQL generated: {response['sql'][:80]}...")
    
except Exception as e:
    print(f"   ❌ Chatbot test failed: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)

# All tests passed!
print("\n" + "=" * 60)
print("✅ All tests passed! Groq is set up correctly!")
print("=" * 60)
print("\n💡 You can now:")
print("   1. Start your API: python run_api.py")
print("   2. Use the chatbot in the frontend")
print("   3. Ask complex questions - Groq will handle them better!")
print()

