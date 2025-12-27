import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, Cell, ReferenceLine
} from 'recharts';
import type { PredictionRow } from '../types';

interface GridVsPredictedChartProps {
  predictions: PredictionRow[];
  showActual?: boolean;
}

export function GridVsPredictedChart({ predictions, showActual = false }: GridVsPredictedChartProps) {
  const data = predictions.slice(0, 15).map((p) => ({
    driver: p.driver.split(' ').pop(),
    fullName: p.driver,
    grid: p.grid_pos,
    predicted: p.pred_pos,
    actual: p.finish_pos,
    team: p.constructor,
  }));

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-white mb-4">Grid vs Predicted Position</h3>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={data} layout="vertical" margin={{ left: 20, right: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#38383F" horizontal={false} />
          <XAxis 
            type="number" 
            stroke="#9CA3AF"
            domain={[1, 20]}
            reversed
            tickFormatter={(v) => `P${v}`}
          />
          <YAxis 
            type="category"
            dataKey="driver" 
            stroke="#9CA3AF"
            width={80}
            tick={{ fill: '#9CA3AF', fontSize: 12 }}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#38383F', 
              border: 'none',
              borderRadius: '8px',
              color: '#fff'
            }}
            formatter={(value) => [`P${value}`, '']}
            labelFormatter={(label) => {
              const item = data.find(d => d.driver === label);
              return item ? `${item.fullName} (${item.team})` : String(label);
            }}
          />
          <Legend />
          <Bar 
            dataKey="grid" 
            fill="#6B7280" 
            name="Grid Position"
            radius={[0, 4, 4, 0]}
          />
          <Bar 
            dataKey="predicted" 
            fill="#E10600" 
            name="Predicted"
            radius={[0, 4, 4, 0]}
          />
          {showActual && (
            <Bar 
              dataKey="actual" 
              fill="#22C55E" 
              name="Actual"
              radius={[0, 4, 4, 0]}
            />
          )}
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-gray-500 mt-2 text-center">
        Lower position number = better (P1 is the winner)
      </p>
    </div>
  );
}

interface PositionDeltaChartProps {
  predictions: PredictionRow[];
}

export function PositionDeltaChart({ predictions }: PositionDeltaChartProps) {
  const data = predictions.slice(0, 15).map((p) => ({
    driver: p.driver.split(' ').pop(),
    fullName: p.driver,
    delta: p.finish_pos !== undefined ? p.pred_pos - p.finish_pos : 0,
    predicted: p.pred_pos,
    actual: p.finish_pos,
  }));

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-white mb-4">Prediction Accuracy (Delta)</h3>
      <ResponsiveContainer width="100%" height={350}>
        <BarChart data={data} layout="vertical" margin={{ left: 20, right: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#38383F" horizontal={false} />
          <XAxis 
            type="number" 
            stroke="#9CA3AF"
            tickFormatter={(v) => v > 0 ? `+${v}` : v.toString()}
          />
          <YAxis 
            type="category"
            dataKey="driver" 
            stroke="#9CA3AF"
            width={80}
            tick={{ fill: '#9CA3AF', fontSize: 12 }}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#38383F', 
              border: 'none',
              borderRadius: '8px',
              color: '#fff'
            }}
            formatter={(value) => {
              const v = Number(value);
              if (v === 0) return ['Perfect prediction!', 'Delta'];
              if (v > 0) return [`Predicted ${v} positions worse`, 'Delta'];
              return [`Predicted ${Math.abs(v)} positions better`, 'Delta'];
            }}
            labelFormatter={(label) => {
              const item = data.find(d => d.driver === label);
              return item ? `${item.fullName} (P${item.predicted} → P${item.actual})` : String(label);
            }}
          />
          <ReferenceLine x={0} stroke="#6B7280" strokeWidth={2} />
          <Bar dataKey="delta" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell 
                key={index} 
                fill={
                  entry.delta === 0 ? '#22C55E' :
                  entry.delta > 0 ? '#EF4444' : '#3B82F6'
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex justify-center gap-6 mt-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-green-500" />
          <span className="text-gray-400">Perfect</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-500" />
          <span className="text-gray-400">Under-predicted</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span className="text-gray-400">Over-predicted</span>
        </div>
      </div>
    </div>
  );
}

// Simple confidence visualization
export function ConfidenceBar({ score, max = 10 }: { score: number; max?: number }) {
  const percentage = Math.min((Math.abs(score) / max) * 100, 100);
  
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
        <div 
          className="h-full bg-gradient-to-r from-f1-red to-orange-500 rounded-full transition-all duration-500"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="text-gray-400 text-xs w-10 text-right">{score.toFixed(1)}</span>
    </div>
  );
}

