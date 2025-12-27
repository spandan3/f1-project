import { useState } from 'react';
import type { PredictionRow } from '../types';

interface PredictionTableProps {
  predictions: PredictionRow[];
  showActual?: boolean;
  isLoading?: boolean;
}

type SortKey = 'pred_pos' | 'grid_pos' | 'driver' | 'constructor' | 'pred_score' | 'finish_pos';
type SortDir = 'asc' | 'desc';

export function PredictionTable({ predictions, showActual = false, isLoading = false }: PredictionTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('pred_pos');
  const [sortDir, setSortDir] = useState<SortDir>('asc');
  const [hoveredRow, setHoveredRow] = useState<number | null>(null);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const sortedPredictions = [...predictions].sort((a, b) => {
    let aVal = a[sortKey];
    let bVal = b[sortKey];
    
    if (typeof aVal === 'string') {
      aVal = aVal.toLowerCase();
      bVal = (bVal as string).toLowerCase();
    }
    
    if (aVal === undefined) return 1;
    if (bVal === undefined) return -1;
    
    if (aVal < bVal) return sortDir === 'asc' ? -1 : 1;
    if (aVal > bVal) return sortDir === 'asc' ? 1 : -1;
    return 0;
  });

  const SortIcon = ({ column }: { column: SortKey }) => (
    <span className={`ml-1 ${sortKey === column ? 'text-f1-red' : 'text-gray-600'}`}>
      {sortKey === column ? (sortDir === 'asc' ? '↑' : '↓') : '↕'}
    </span>
  );

  const getRowClass = (pos: number) => {
    if (pos === 1) return 'podium-gold';
    if (pos === 2) return 'podium-silver';
    if (pos === 3) return 'podium-bronze';
    return '';
  };

  const getPositionChange = (gridPos: number, predPos: number) => {
    const change = gridPos - predPos;
    if (change > 0) return { text: `+${change}`, class: 'text-green-400' };
    if (change < 0) return { text: `${change}`, class: 'text-red-400' };
    return { text: '—', class: 'text-gray-500' };
  };

  if (isLoading) {
    return <PredictionTableSkeleton />;
  }

  return (
    <div className="card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-600">
              <th 
                className="text-left py-4 px-4 text-gray-400 font-semibold cursor-pointer hover:text-white transition-colors"
                onClick={() => handleSort('pred_pos')}
              >
                Pos <SortIcon column="pred_pos" />
              </th>
              <th 
                className="text-left py-4 px-4 text-gray-400 font-semibold cursor-pointer hover:text-white transition-colors"
                onClick={() => handleSort('driver')}
              >
                Driver <SortIcon column="driver" />
              </th>
              <th 
                className="text-left py-4 px-4 text-gray-400 font-semibold cursor-pointer hover:text-white transition-colors hidden md:table-cell"
                onClick={() => handleSort('constructor')}
              >
                Team <SortIcon column="constructor" />
              </th>
              <th 
                className="text-left py-4 px-4 text-gray-400 font-semibold cursor-pointer hover:text-white transition-colors"
                onClick={() => handleSort('grid_pos')}
              >
                Grid <SortIcon column="grid_pos" />
              </th>
              <th className="text-left py-4 px-4 text-gray-400 font-semibold hidden sm:table-cell">
                Change
              </th>
              <th 
                className="text-left py-4 px-4 text-gray-400 font-semibold cursor-pointer hover:text-white transition-colors hidden lg:table-cell"
                onClick={() => handleSort('pred_score')}
              >
                Score <SortIcon column="pred_score" />
              </th>
              {showActual && (
                <th 
                  className="text-left py-4 px-4 text-gray-400 font-semibold cursor-pointer hover:text-white transition-colors"
                  onClick={() => handleSort('finish_pos')}
                >
                  Actual <SortIcon column="finish_pos" />
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {sortedPredictions.map((pred, idx) => {
              const posChange = getPositionChange(pred.grid_pos, pred.pred_pos);
              const isPodium = pred.pred_pos <= 3;
              const isHovered = hoveredRow === idx;
              
              return (
                <tr 
                  key={`${pred.driver}-${idx}`}
                  className={`
                    table-row animate-fade-in-up
                    ${getRowClass(pred.pred_pos)}
                    ${isHovered ? 'bg-white/10' : ''}
                  `}
                  style={{ animationDelay: `${idx * 0.03}s` }}
                  onMouseEnter={() => setHoveredRow(idx)}
                  onMouseLeave={() => setHoveredRow(null)}
                >
                  <td className="py-4 px-4">
                    <span className={`
                      font-racing font-bold text-lg
                      ${isPodium ? 'text-f1-red' : 'text-white'}
                    `}>
                      P{pred.pred_pos}
                    </span>
                  </td>
                  <td className="py-4 px-4">
                    <div className="flex items-center gap-3">
                      <div className={`
                        w-1 h-8 rounded-full
                        ${pred.pred_pos === 1 ? 'bg-yellow-400' : 
                          pred.pred_pos === 2 ? 'bg-gray-400' : 
                          pred.pred_pos === 3 ? 'bg-orange-500' : 'bg-gray-600'}
                      `} />
                      <div>
                        <div className="text-white font-medium">{pred.driver}</div>
                        <div className="text-gray-400 text-sm md:hidden">{pred.constructor}</div>
                      </div>
                    </div>
                  </td>
                  <td className="py-4 px-4 text-gray-300 hidden md:table-cell">
                    {pred.constructor}
                  </td>
                  <td className="py-4 px-4 text-gray-400">
                    P{pred.grid_pos}
                  </td>
                  <td className={`py-4 px-4 font-medium hidden sm:table-cell ${posChange.class}`}>
                    {posChange.text}
                  </td>
                  <td className="py-4 px-4 hidden lg:table-cell">
                    <div className="flex items-center gap-2">
                      <div className="w-16 bg-gray-700 rounded-full h-1.5">
                        <div
                          className="bg-f1-red h-1.5 rounded-full transition-all duration-500"
                          style={{ width: `${Math.min(Math.abs(pred.pred_score) * 10, 100)}%` }}
                        />
                      </div>
                      <span className="text-gray-400 text-sm w-12">
                        {pred.pred_score.toFixed(2)}
                      </span>
                    </div>
                  </td>
                  {showActual && (
                    <td className="py-4 px-4">
                      {pred.finish_pos ? (
                        <span className={`
                          font-semibold
                          ${pred.finish_pos === pred.pred_pos ? 'text-green-400' : 
                            Math.abs((pred.finish_pos || 0) - pred.pred_pos) <= 2 ? 'text-yellow-400' : 'text-gray-400'}
                        `}>
                          P{pred.finish_pos}
                          {pred.finish_pos === pred.pred_pos && ' ✓'}
                        </span>
                      ) : (
                        <span className="text-gray-600">—</span>
                      )}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PredictionTableSkeleton() {
  return (
    <div className="card overflow-hidden">
      <div className="space-y-4 p-4">
        {Array.from({ length: 10 }).map((_, i) => (
          <div key={i} className="flex items-center gap-4">
            <div className="skeleton h-6 w-12 rounded" />
            <div className="skeleton h-6 w-32 rounded" />
            <div className="skeleton h-6 w-24 rounded hidden md:block" />
            <div className="skeleton h-6 w-12 rounded" />
            <div className="skeleton h-6 w-16 rounded hidden lg:block" />
          </div>
        ))}
      </div>
    </div>
  );
}

