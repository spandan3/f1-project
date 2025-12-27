import type { PredictionRow } from '../types';

interface PodiumCardProps {
  predictions: PredictionRow[];
  showActual?: boolean;
}

export function PodiumCard({ predictions, showActual = false }: PodiumCardProps) {
  const podium = predictions.slice(0, 3);
  
  // Reorder for display: 2nd, 1st, 3rd (visual podium style - 1st in center, tallest)
  const displayOrder = podium.length >= 3 ? [podium[1], podium[0], podium[2]] : podium;
  // Arrays match display order: [2nd/silver, 1st/gold, 3rd/bronze]
  const heights = ['h-28', 'h-36', 'h-24'];
  const positions = ['2nd', '1st', '3rd'];
  const colors = [
    'from-gray-400 to-gray-500',    // Silver for 2nd
    'from-yellow-400 to-yellow-600', // Gold for 1st
    'from-orange-600 to-orange-700', // Bronze for 3rd
  ];
  const textColors = ['text-gray-300', 'text-yellow-400', 'text-orange-500'];
  const borderColors = ['border-gray-400', 'border-yellow-400', 'border-orange-500'];

  if (podium.length === 0) {
    return null;
  }

  return (
    <div className="card-glass">
      <h3 className="text-lg font-semibold text-white mb-6 text-center font-racing">
        Predicted Podium
      </h3>
      
      <div className="flex items-end justify-center gap-4">
        {displayOrder.map((pred, idx) => {
          // idx directly maps to arrays since both are in display order [2nd, 1st, 3rd]
          const isCorrect = showActual && pred.finish_pos === pred.pred_pos;
          
          return (
            <div 
              key={pred.driver}
              className="flex flex-col items-center animate-fade-in-up"
              style={{ animationDelay: `${idx * 0.15}s` }}
            >
              {/* Driver info */}
              <div className="text-center mb-2">
                <div className={`text-xs font-bold ${textColors[idx]} mb-1`}>
                  P{pred.pred_pos}
                </div>
                <div className="text-white font-semibold text-sm truncate max-w-[100px]">
                  {pred.driver.split(' ').pop()}
                </div>
                <div className="text-gray-400 text-xs truncate max-w-[100px]">
                  {pred.constructor}
                </div>
                {showActual && pred.finish_pos && (
                  <div className={`text-xs mt-1 ${isCorrect ? 'text-green-400' : 'text-gray-500'}`}>
                    Actual: P{pred.finish_pos}
                    {isCorrect && ' ✓'}
                  </div>
                )}
              </div>
              
              {/* Podium block */}
              <div 
                className={`
                  ${heights[idx]} w-20 rounded-t-lg 
                  bg-gradient-to-t ${colors[idx]}
                  border-t-4 ${borderColors[idx]}
                  flex items-end justify-center pb-2
                  shadow-lg transition-all duration-300 hover:scale-105
                `}
              >
                <span className="text-2xl font-racing font-bold text-white drop-shadow-lg">
                  {positions[idx]}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Mini version for compact display
export function PodiumMini({ predictions }: { predictions: PredictionRow[] }) {
  const podium = predictions.slice(0, 3);
  const medals = ['🥇', '🥈', '🥉'];

  return (
    <div className="flex items-center gap-4">
      {podium.map((pred, idx) => (
        <div key={pred.driver} className="flex items-center gap-2">
          <span className="text-lg">{medals[idx]}</span>
          <span className="text-white text-sm font-medium">
            {pred.driver.split(' ').pop()}
          </span>
        </div>
      ))}
    </div>
  );
}

