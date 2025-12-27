interface LoadingSpinnerProps {
  message?: string;
  size?: 'sm' | 'md' | 'lg';
}

export function LoadingSpinner({ message = 'Loading...', size = 'md' }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'h-6 w-6 border-2',
    md: 'h-12 w-12 border-3',
    lg: 'h-16 w-16 border-4',
  };

  return (
    <div className="flex flex-col items-center justify-center p-8">
      <div className="relative">
        {/* Outer ring */}
        <div className={`
          ${sizeClasses[size]} 
          rounded-full border-gray-700 animate-pulse
        `} />
        {/* Spinning ring */}
        <div className={`
          absolute inset-0 ${sizeClasses[size]} 
          rounded-full border-transparent border-t-f1-red animate-spin
        `} />
      </div>
      <p className="mt-4 text-gray-400 animate-pulse">{message}</p>
    </div>
  );
}

// Skeleton components for loading states
export function SkeletonLine({ className = '' }: { className?: string }) {
  return <div className={`skeleton h-4 rounded ${className}`} />;
}

export function SkeletonCard() {
  return (
    <div className="card space-y-4 animate-pulse">
      <SkeletonLine className="w-1/3 h-6" />
      <SkeletonLine className="w-full" />
      <SkeletonLine className="w-2/3" />
      <SkeletonLine className="w-1/2" />
    </div>
  );
}

export function SkeletonTable({ rows = 5 }: { rows?: number }) {
  return (
    <div className="card overflow-hidden">
      <div className="space-y-4 p-4">
        <div className="flex gap-4">
          <SkeletonLine className="w-16" />
          <SkeletonLine className="w-32" />
          <SkeletonLine className="w-24 hidden md:block" />
          <SkeletonLine className="w-16" />
        </div>
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="flex gap-4">
            <SkeletonLine className="w-16" />
            <SkeletonLine className="w-32" />
            <SkeletonLine className="w-24 hidden md:block" />
            <SkeletonLine className="w-16" />
          </div>
        ))}
      </div>
    </div>
  );
}
