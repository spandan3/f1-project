interface ErrorMessageProps {
  message: string;
  onRetry?: () => void;
}

export function ErrorMessage({ message, onRetry }: ErrorMessageProps) {
  return (
    <div className="bg-red-900/20 border border-red-500/50 rounded-xl p-6 animate-fade-in">
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0 w-10 h-10 bg-red-500/20 rounded-full flex items-center justify-center">
          <svg 
            className="w-5 h-5 text-red-400" 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" 
            />
          </svg>
        </div>
        <div className="flex-1">
          <h3 className="text-red-400 font-semibold mb-1">Something went wrong</h3>
          <p className="text-red-300/80 text-sm">{message}</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-4 text-red-400 hover:text-red-300 text-sm font-medium 
                       flex items-center gap-2 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                      d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Try again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// Empty state component
export function EmptyState({ 
  icon = '📭', 
  title, 
  description,
  action,
  actionLabel,
}: { 
  icon?: string; 
  title: string; 
  description?: string;
  action?: () => void;
  actionLabel?: string;
}) {
  return (
    <div className="card-glass text-center py-12 px-6">
      <div className="text-5xl mb-4">{icon}</div>
      <h3 className="text-xl font-semibold text-white mb-2">{title}</h3>
      {description && (
        <p className="text-gray-400 max-w-md mx-auto">{description}</p>
      )}
      {action && actionLabel && (
        <button onClick={action} className="btn-primary mt-6">
          {actionLabel}
        </button>
      )}
    </div>
  );
}
