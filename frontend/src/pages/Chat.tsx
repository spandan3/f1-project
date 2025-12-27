import { useState } from 'react';
import type { ChatMessage } from '../types';

export function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      role: 'assistant',
      content: 'Hello! I\'m the F1 Predictor Assistant. This feature is coming soon! In the future, you\'ll be able to ask me questions like:\n\n• "Who had the fastest lap in Monaco 2023?"\n• "Compare Verstappen and Hamilton\'s form this season"\n• "What\'s the predicted podium for the next race?"',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (!input.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    const botResponse: ChatMessage = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: 'Thanks for your message! The chat assistant is currently under development. Check back soon for full RAG-powered Q&A capabilities.',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage, botResponse]);
    setInput('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <h1 className="font-racing text-3xl text-white">
            Chat Assistant
          </h1>
          <span className="bg-yellow-500 text-black px-2 py-1 rounded text-xs font-bold">
            Coming Soon
          </span>
        </div>
        <p className="text-gray-400">
          Ask questions about F1 data, predictions, and driver performance
        </p>
      </div>

      {/* Chat Container */}
      <div className="card h-[600px] flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`
                  max-w-[80%] p-4 rounded-2xl animate-fade-in
                  ${message.role === 'user'
                    ? 'bg-f1-red text-white rounded-br-md'
                    : 'bg-f1-dark text-gray-200 rounded-bl-md border border-gray-700'}
                `}
              >
                {message.role === 'assistant' && (
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-f1-red">🏎️</span>
                    <span className="text-xs text-gray-400 font-medium">F1 Predictor</span>
                  </div>
                )}
                <p className="whitespace-pre-wrap text-sm">{message.content}</p>
                <div className={`text-xs mt-2 ${
                  message.role === 'user' ? 'text-white/60' : 'text-gray-500'
                }`}>
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-700 p-4">
          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about F1 predictions, drivers, or races..."
              className="input flex-1"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim()}
              className="btn-primary px-6 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Send
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Press Enter to send • Shift+Enter for new line
          </p>
        </div>
      </div>

      {/* Feature Preview */}
      <div className="mt-8 grid md:grid-cols-3 gap-4">
        <FeaturePreview
          icon="🔍"
          title="Data Queries"
          description="Ask about lap times, positions, and historical stats"
        />
        <FeaturePreview
          icon="📊"
          title="Predictions"
          description="Get predictions for upcoming races and compare drivers"
        />
        <FeaturePreview
          icon="📈"
          title="Analysis"
          description="Deep dive into performance trends and strategy insights"
        />
      </div>

      {/* Coming Soon Banner */}
      <div className="mt-8 card-glass text-center py-8 racing-stripes">
        <h3 className="text-xl font-semibold text-white mb-2">
          RAG-Powered Assistant Coming Soon
        </h3>
        <p className="text-gray-400 max-w-md mx-auto">
          We're building a retrieval-augmented generation system that will let you 
          query our entire F1 database using natural language.
        </p>
      </div>
    </div>
  );
}

function FeaturePreview({ icon, title, description }: { 
  icon: string; 
  title: string; 
  description: string; 
}) {
  return (
    <div className="card-hover opacity-60">
      <div className="text-2xl mb-2">{icon}</div>
      <h4 className="font-semibold text-white mb-1">{title}</h4>
      <p className="text-sm text-gray-400">{description}</p>
      <div className="text-xs text-yellow-500 mt-2">Coming Soon</div>
    </div>
  );
}

