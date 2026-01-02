import { useState } from 'react';
import { apiClient } from '../lib/api';
import { LoadingSpinner } from '../components/LoadingSpinner';
import type { ChatMessage } from '../types';

export function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      role: 'assistant',
      content: 'Hello! I\'m your F1 Database Assistant. I can answer questions about our Formula 1 database covering 2018-2025 seasons.\n\nAsk me anything about:\n• Race results and positions\n• Championship standings\n• Lap times and fastest laps\n• Driver and team statistics\n• Qualifying and pole positions\n\nTry questions like:\n• "Who won the 2023 championship?"\n• "Who finished 2nd in Monaco 2024?"\n• "Who had the fastest lap in Bahrain 2023?"\n• "How many podiums did Verstappen get in 2024?"',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await apiClient.askChatbot(input);
      
      const botResponse: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.answer,
        timestamp: new Date(),
        rows: response.rows,
        sql: response.sql,
        explanation: response.explanation,
        method: response.method,
      };

      setMessages(prev => [...prev, botResponse]);
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}. Please try rephrasing your question.`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
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
            F1 Database Assistant
          </h1>
          <span className="bg-green-500 text-black px-2 py-1 rounded text-xs font-bold">
            Live
          </span>
        </div>
        <p className="text-gray-400">
          Ask questions about our F1 database (2018-2025). Get instant answers about race results, championship standings, lap times, and more.
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
                    {message.method === 'llm' && (
                      <span className="text-xs px-2 py-0.5 rounded bg-blue-500/20 text-blue-400">
                        AI
                      </span>
                    )}
                  </div>
                )}
                <p className="whitespace-pre-wrap text-sm">{message.content}</p>
                
                {/* Display data table if rows exist */}
                {message.rows && message.rows.length > 0 && (
                  <div className="mt-3 overflow-x-auto">
                    <table className="w-full text-xs border-collapse">
                      <thead>
                        <tr className="border-b border-gray-600">
                          {Object.keys(message.rows[0]).map((key) => (
                            <th key={key} className="text-left p-2 text-gray-400 font-semibold">
                              {key}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {message.rows.slice(0, 10).map((row, idx) => (
                          <tr key={idx} className="border-b border-gray-700/50 hover:bg-gray-800/50">
                            {Object.values(row).map((val, colIdx) => (
                              <td key={colIdx} className="p-2 text-gray-300">
                                {val !== null && val !== undefined ? String(val) : 'N/A'}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {message.rows.length > 10 && (
                      <p className="text-xs text-gray-500 mt-2">
                        Showing 10 of {message.rows.length} results
                      </p>
                    )}
                  </div>
                )}
                
                {/* Show SQL query in expandable section */}
                {message.sql && (
                  <details className="mt-3">
                    <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
                      View SQL Query
                    </summary>
                    <pre className="mt-2 p-2 bg-gray-900 rounded text-xs text-gray-400 overflow-x-auto">
                      {message.sql}
                    </pre>
                  </details>
                )}
                
                <div className={`text-xs mt-2 ${
                  message.role === 'user' ? 'text-white/60' : 'text-gray-500'
                }`}>
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-f1-dark/50 text-gray-300 rounded-bl-md border border-gray-700/50 p-3 max-w-[60%]">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-gray-600 border-t-f1-red rounded-full animate-spin" />
                  <span className="text-xs text-gray-400">Thinking...</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-700 p-4">
          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask questions about our F1 database (e.g., 'Who won the 2023 championship?')"
              className="input flex-1"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="btn-primary px-6 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Sending...' : 'Send'}
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Press Enter to send
          </p>
        </div>
      </div>

      {/* Feature Preview */}
      <div className="mt-8 grid md:grid-cols-3 gap-4">
        <FeaturePreview
          icon="🏁"
          title="Race Results"
          description="Query race winners, positions, and finishing orders"
        />
        <FeaturePreview
          icon="🏆"
          title="Championship Data"
          description="Ask about championship standings, points, and rankings"
        />
        <FeaturePreview
          icon="⏱️"
          title="Lap Times & Stats"
          description="Get fastest laps, sector times, and performance metrics"
        />
      </div>

      {/* Info Banner */}
      <div className="mt-8 card-glass text-center py-6">
        <h3 className="text-lg font-semibold text-white mb-2">
          💡 Database Coverage
        </h3>
        <div className="text-gray-400 max-w-2xl mx-auto text-sm space-y-1">
          <p>• <strong>Years:</strong> 2018-2025 seasons (173 races, 3,978 driver-race records)</p>
          <p>• <strong>Data:</strong> Race results, qualifying, lap times, weather conditions</p>
          <p>• <strong>Tip:</strong> Include the year in your question for best results (e.g., "in 2023")</p>
        </div>
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
      <div className="card-hover">
        <div className="text-2xl mb-2">{icon}</div>
        <h4 className="font-semibold text-white mb-1">{title}</h4>
        <p className="text-sm text-gray-400">{description}</p>
      </div>
  );
}

