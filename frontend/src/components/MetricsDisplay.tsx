import type { Metrics } from '../types';

interface MetricsDisplayProps {
  metrics: Metrics;
  compact?: boolean;
}

export function MetricsDisplay({ metrics, compact = false }: MetricsDisplayProps) {
  const items = [
    { 
      label: 'NDCG@3', 
      value: metrics.ndcg_3, 
      format: (v: number) => v.toFixed(3),
      description: 'Ranking accuracy (top 3)',
      color: getMetricColor(metrics.ndcg_3),
    },
    { 
      label: 'NDCG@10', 
      value: metrics.ndcg_10, 
      format: (v: number) => v.toFixed(3),
      description: 'Ranking accuracy (top 10)',
      color: getMetricColor(metrics.ndcg_10),
    },
    { 
      label: 'Top-3 Hit', 
      value: metrics.top3_hit, 
      format: (v: number) => `${(v * 100).toFixed(0)}%`,
      description: 'Podium prediction rate',
      color: getMetricColor(metrics.top3_hit),
    },
    { 
      label: 'Spearman ρ', 
      value: metrics.spearman_rho, 
      format: (v: number) => v.toFixed(3),
      description: 'Rank correlation',
      color: getMetricColor(metrics.spearman_rho),
    },
  ];

  if (compact) {
    return (
      <div className="flex flex-wrap gap-4">
        {items.map(item => (
          <div key={item.label} className="flex items-center gap-2">
            <span className="text-gray-400 text-sm">{item.label}:</span>
            <span className={`font-semibold ${item.color}`}>
              {item.format(item.value)}
            </span>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {items.map((item, idx) => (
        <div 
          key={item.label}
          className="metric-card animate-fade-in-up"
          style={{ animationDelay: `${idx * 0.1}s` }}
        >
          <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">
            {item.label}
          </div>
          <div className={`text-2xl font-bold ${item.color}`}>
            {item.format(item.value)}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {item.description}
          </div>
          <div className="mt-2 h-1 bg-gray-700 rounded-full overflow-hidden">
            <div 
              className={`h-full rounded-full transition-all duration-1000 ${getBarColor(item.value)}`}
              style={{ width: `${Math.min(item.value * 100, 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function getMetricColor(value: number): string {
  if (value >= 0.7) return 'text-green-400';
  if (value >= 0.5) return 'text-yellow-400';
  if (value >= 0.3) return 'text-orange-400';
  return 'text-red-400';
}

function getBarColor(value: number): string {
  if (value >= 0.7) return 'bg-green-500';
  if (value >= 0.5) return 'bg-yellow-500';
  if (value >= 0.3) return 'bg-orange-500';
  return 'bg-red-500';
}

// Simple inline metric display
export function MetricBadge({ label, value, suffix = '' }: { label: string; value: number; suffix?: string }) {
  const color = getMetricColor(value);
  return (
    <div className="inline-flex items-center gap-1 bg-f1-dark/50 rounded-lg px-3 py-1">
      <span className="text-gray-400 text-sm">{label}</span>
      <span className={`font-bold ${color}`}>{value.toFixed(3)}{suffix}</span>
    </div>
  );
}

