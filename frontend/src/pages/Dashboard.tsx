import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../lib/api';
import { RaceSelectorCompact } from '../components/RaceSelector';
import { PredictionTable } from '../components/PredictionTable';
import { PodiumCard } from '../components/PodiumCard';
import { ComparisonTable } from '../components/ComparisonTable';
import { MetricsDisplay } from '../components/MetricsDisplay';
import { GridVsPredictedChart, PositionDeltaChart } from '../components/Charts';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorMessage } from '../components/ErrorMessage';
import type { Race } from '../types';

export function Dashboard() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedYear, setSelectedYear] = useState(() => 
    Number(searchParams.get('year')) || 2024
  );
  const [selectedRound, setSelectedRound] = useState(() => 
    Number(searchParams.get('round')) || 1
  );
  const [activeTab, setActiveTab] = useState<'predictions' | 'analysis'>('predictions');

  // Sync URL params
  useEffect(() => {
    setSearchParams({ year: String(selectedYear), round: String(selectedRound) });
  }, [selectedYear, selectedRound, setSearchParams]);

  const { data: status } = useQuery({
    queryKey: ['status'],
    queryFn: () => apiClient.getStatus(),
  });

  const { data: races } = useQuery({
    queryKey: ['races', selectedYear],
    queryFn: () => apiClient.getRaces(selectedYear),
    enabled: selectedYear > 0,
  });

  const {
    data: predictions,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['predictions', selectedYear, selectedRound],
    queryFn: () => apiClient.getPredictions(selectedYear, selectedRound),
    enabled: selectedYear > 0 && selectedRound > 0,
  });

  const selectedRace = races?.find((r: Race) => r.round === selectedRound);
  const hasActualResults = predictions?.predictions?.some(p => p.finish_pos !== undefined);

  const handleYearChange = (year: number) => {
    setSelectedYear(year);
    setSelectedRound(1);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 mb-8">
        <div>
          <h1 className="font-racing text-3xl text-white mb-2">
            Race Dashboard
          </h1>
          <p className="text-gray-400">
            View predictions and analysis for any race
          </p>
        </div>

        <RaceSelectorCompact
          selectedYear={selectedYear}
          selectedRound={selectedRound}
          onYearChange={handleYearChange}
          onRoundChange={setSelectedRound}
          availableYears={status?.available_years || [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]}
        />
      </div>

      {/* Race Info Banner */}
      {selectedRace && (
        <div className="card-glass mb-8 animate-fade-in">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <div className="flex items-center gap-3">
                <span className="bg-f1-red text-white px-3 py-1 rounded font-racing font-bold">
                  R{selectedRound}
                </span>
                <h2 className="text-xl sm:text-2xl font-bold text-white">
                  {selectedRace.race_name}
                </h2>
              </div>
              <div className="text-gray-400 mt-1">
                {selectedYear} Season • {selectedRace.date}
              </div>
            </div>
            <div className="flex items-center gap-2">
              {hasActualResults ? (
                <span className="chip-success">Results Available</span>
              ) : (
                <span className="chip-info">Pre-Race Prediction</span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="py-16">
          <LoadingSpinner message="Loading predictions..." />
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="py-8">
          <ErrorMessage message={(error as Error).message} />
          <button onClick={() => refetch()} className="btn-secondary mt-4">
            Try Again
          </button>
        </div>
      )}

      {/* Content */}
      {predictions && !isLoading && (
        <>
          {/* Metrics */}
          {predictions.metrics && (
            <div className="mb-8 animate-fade-in-up">
              <h3 className="text-lg font-semibold text-white mb-4">Model Performance</h3>
              <MetricsDisplay metrics={predictions.metrics} />
            </div>
          )}

          {/* Tabs */}
          <div className="flex gap-2 mb-6 border-b border-gray-700">
            <button
              onClick={() => setActiveTab('predictions')}
              className={`px-4 py-3 font-medium transition-colors relative ${
                activeTab === 'predictions' 
                  ? 'text-white' 
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Predictions
              {activeTab === 'predictions' && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-f1-red" />
              )}
            </button>
            <button
              onClick={() => setActiveTab('analysis')}
              className={`px-4 py-3 font-medium transition-colors relative ${
                activeTab === 'analysis' 
                  ? 'text-white' 
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Analysis
              {hasActualResults && (
                <span className="ml-2 w-2 h-2 bg-green-400 rounded-full inline-block" />
              )}
              {activeTab === 'analysis' && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-f1-red" />
              )}
            </button>
          </div>

          {/* Tab Content */}
          {activeTab === 'predictions' && (
            <div className="space-y-8 animate-fade-in">
              {/* Podium */}
              <PodiumCard 
                predictions={predictions.predictions} 
                showActual={hasActualResults}
              />

              {/* Grid vs Predicted Chart */}
              <GridVsPredictedChart 
                predictions={predictions.predictions}
                showActual={hasActualResults}
              />

              {/* Full Predictions Table */}
              <div>
                <h3 className="text-lg font-semibold text-white mb-4">
                  Full Grid Predictions
                </h3>
                <PredictionTable 
                  predictions={predictions.predictions}
                  showActual={hasActualResults}
                />
              </div>
            </div>
          )}

          {activeTab === 'analysis' && (
            <div className="space-y-8 animate-fade-in">
              {hasActualResults ? (
                <>
                  {/* Comparison Table */}
                  <ComparisonTable predictions={predictions.predictions} />

                  {/* Position Delta Chart */}
                  <PositionDeltaChart predictions={predictions.predictions} />
                </>
              ) : (
                <div className="card-glass text-center py-16">
                  <div className="text-5xl mb-4">🏁</div>
                  <h3 className="text-xl font-semibold text-white mb-2">
                    Race Not Yet Complete
                  </h3>
                  <p className="text-gray-400 max-w-md mx-auto">
                    Analysis will be available after the race is completed and 
                    results are updated in the system.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Footer Note */}
          <div className="mt-8 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
            <p className="text-sm text-blue-300">
              <strong>Note:</strong> Predictions are generated using only pre-race data 
              (qualifying results, driver form, track history). Actual race outcomes may 
              differ due to race-day incidents, strategy, and other factors.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
