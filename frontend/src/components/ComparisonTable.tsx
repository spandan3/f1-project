import type { PredictionRow } from '../types';

interface ComparisonTableProps {
  predictions: PredictionRow[];
}

export function ComparisonTable({ predictions }: ComparisonTableProps) {
  // Only show if we have actual results
  const hasActual = predictions.some(p => p.finish_pos !== undefined);
  
  if (!hasActual) {
    return (
      <div className="card-glass text-center py-8">
        <p className="text-gray-400">
          Actual results not yet available for this race.
        </p>
        <p className="text-sm text-gray-500 mt-2">
          Results will appear after the race is completed.
        </p>
      </div>
    );
  }

  const getAccuracyStats = () => {
    let exactMatches = 0;
    let closeMatches = 0; // within 2 positions
    let podiumHits = 0;
    let totalDelta = 0;

    predictions.forEach(pred => {
      if (pred.finish_pos !== undefined) {
        const delta = Math.abs(pred.pred_pos - pred.finish_pos);
        totalDelta += delta;
        
        if (delta === 0) exactMatches++;
        if (delta <= 2) closeMatches++;
        
        // Check podium hits
        if (pred.pred_pos <= 3 && pred.finish_pos <= 3) {
          podiumHits++;
        }
      }
    });

    return {
      exactMatches,
      closeMatches,
      podiumHits,
      avgDelta: totalDelta / predictions.length,
    };
  };

  const stats = getAccuracyStats();

  return (
    <div className="space-y-6">
      {/* Stats Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="metric-card text-center">
          <div className="text-3xl font-bold text-green-400">{stats.exactMatches}</div>
          <div className="text-sm text-gray-400">Exact Matches</div>
        </div>
        <div className="metric-card text-center">
          <div className="text-3xl font-bold text-yellow-400">{stats.closeMatches}</div>
          <div className="text-sm text-gray-400">Within 2 Pos</div>
        </div>
        <div className="metric-card text-center">
          <div className="text-3xl font-bold text-f1-red">{stats.podiumHits}/3</div>
          <div className="text-sm text-gray-400">Podium Hits</div>
        </div>
        <div className="metric-card text-center">
          <div className="text-3xl font-bold text-blue-400">{stats.avgDelta.toFixed(1)}</div>
          <div className="text-sm text-gray-400">Avg Delta</div>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="card overflow-hidden">
        <h3 className="text-lg font-semibold text-white mb-4">Predicted vs Actual</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-600">
                <th className="text-left py-3 px-4 text-gray-400 font-semibold">Driver</th>
                <th className="text-center py-3 px-4 text-gray-400 font-semibold">Predicted</th>
                <th className="text-center py-3 px-4 text-gray-400 font-semibold">Actual</th>
                <th className="text-center py-3 px-4 text-gray-400 font-semibold">Delta</th>
                <th className="text-center py-3 px-4 text-gray-400 font-semibold hidden sm:table-cell">Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((pred, idx) => {
                const delta = pred.finish_pos !== undefined 
                  ? pred.finish_pos - pred.pred_pos 
                  : null;
                const isExact = delta === 0;
                const isClose = delta !== null && Math.abs(delta) <= 2;
                const isPodiumHit = pred.pred_pos <= 3 && (pred.finish_pos || 0) <= 3;
                
                return (
                  <tr 
                    key={`${pred.driver}-${idx}`}
                    className={`
                      table-row animate-slide-in
                      ${isExact ? 'bg-green-500/10' : isClose ? 'bg-yellow-500/5' : ''}
                    `}
                    style={{ animationDelay: `${idx * 0.03}s` }}
                  >
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        {isPodiumHit && (
                          <span className="text-yellow-400">🏆</span>
                        )}
                        <span className="text-white font-medium">{pred.driver}</span>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-center">
                      <span className={`
                        font-bold
                        ${pred.pred_pos <= 3 ? 'text-f1-red' : 'text-gray-300'}
                      `}>
                        P{pred.pred_pos}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-center">
                      {pred.finish_pos !== undefined ? (
                        <span className={`
                          font-bold
                          ${pred.finish_pos <= 3 ? 'text-yellow-400' : 'text-gray-300'}
                        `}>
                          P{pred.finish_pos}
                        </span>
                      ) : (
                        <span className="text-gray-600">—</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-center">
                      {delta !== null ? (
                        <span className={`
                          font-semibold
                          ${delta === 0 ? 'text-green-400' : 
                            delta > 0 ? 'text-red-400' : 'text-blue-400'}
                        `}>
                          {delta === 0 ? '✓' : delta > 0 ? `+${delta}` : delta}
                        </span>
                      ) : (
                        <span className="text-gray-600">—</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-center hidden sm:table-cell">
                      {delta !== null && (
                        <div className="flex justify-center">
                          {isExact ? (
                            <span className="chip-success text-xs">Perfect</span>
                          ) : isClose ? (
                            <span className="chip-warning text-xs">Close</span>
                          ) : (
                            <span className="chip-error text-xs">Off</span>
                          )}
                        </div>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

