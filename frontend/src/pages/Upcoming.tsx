import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { apiClient } from '../lib/api';
import { PredictionTable } from '../components/PredictionTable';
import { PodiumCard } from '../components/PodiumCard';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorMessage } from '../components/ErrorMessage';
import type { Race } from '../types';

export function Upcoming() {
  const navigate = useNavigate();
  const [selectedYear] = useState(2026);
  const [selectedRound, setSelectedRound] = useState(1);
  const [predictionRequested, setPredictionRequested] = useState(false);

  const { data: races, isLoading: racesLoading } = useQuery({
    queryKey: ['races', selectedYear],
    queryFn: () => apiClient.getRaces(selectedYear),
  });

  const {
    data: predictions,
    isLoading: predictionsLoading,
    error: predictionsError,
  } = useQuery({
    queryKey: ['predictions', selectedYear, selectedRound],
    queryFn: () => apiClient.getPredictions(selectedYear, selectedRound),
    enabled: predictionRequested,
  });

  const selectedRace = races?.find((r: Race) => r.round === selectedRound);

  const handleGeneratePrediction = () => {
    setPredictionRequested(true);
  };

  // Find next upcoming race
  const getUpcomingRaces = () => {
    if (!races) return [];
    const today = new Date();
    return races.filter((race: Race) => new Date(race.date) >= today);
  };

  const upcomingRaces = getUpcomingRaces();

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <span className="bg-f1-red text-white px-3 py-1 rounded font-racing font-bold">
            2026
          </span>
          <h1 className="font-racing text-3xl text-white">
            Upcoming Races
          </h1>
        </div>
        <p className="text-gray-400">
          Generate pre-race predictions for upcoming F1 events
        </p>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Race Selection */}
        <div className="lg:col-span-1 space-y-6">
          {/* Race List */}
          <div className="card">
            <h2 className="text-lg font-semibold text-white mb-4">
              Select Race
            </h2>

            {racesLoading ? (
              <div className="space-y-3">
                {Array.from({ length: 5 }).map((_, i) => (
                  <div key={i} className="skeleton h-16 rounded-lg" />
                ))}
              </div>
            ) : races && races.length > 0 ? (
              <div className="space-y-2 max-h-[500px] overflow-y-auto pr-2">
                {races.map((race: Race) => {
                  const isUpcoming = new Date(race.date) >= new Date();
                  const isSelected = race.round === selectedRound;
                  
                  return (
                    <button
                      key={race.round}
                      onClick={() => {
                        setSelectedRound(race.round);
                        setPredictionRequested(false);
                      }}
                      className={`
                        w-full text-left p-4 rounded-lg transition-all duration-200
                        ${isSelected 
                          ? 'bg-f1-red text-white' 
                          : 'bg-f1-dark/50 hover:bg-white/5 text-gray-300'}
                        ${!isUpcoming && !isSelected ? 'opacity-50' : ''}
                      `}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium flex items-center gap-2">
                            R{race.round} — {race.race_name}
                            {isUpcoming && (
                              <span className={`text-xs px-2 py-0.5 rounded ${
                                isSelected ? 'bg-white/20' : 'bg-green-500/20 text-green-400'
                              }`}>
                                Upcoming
                              </span>
                            )}
                          </div>
                          <div className={`text-sm ${isSelected ? 'text-white/80' : 'text-gray-500'}`}>
                            {race.date}
                          </div>
                        </div>
                        <span className={`text-2xl font-racing ${isSelected ? 'text-white' : 'text-gray-600'}`}>
                          {race.round}
                        </span>
                      </div>
                    </button>
                  );
                })}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-400">
                <p>No races available for 2026 yet.</p>
                <p className="text-sm mt-2">Calendar will be updated when released.</p>
              </div>
            )}
          </div>

          {/* Quick Info */}
          <div className="card-glass">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
              About Pre-Race Predictions
            </h3>
            <ul className="text-sm text-gray-400 space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-f1-red">•</span>
                Uses qualifying results when available
              </li>
              <li className="flex items-start gap-2">
                <span className="text-f1-red">•</span>
                Considers driver & team form
              </li>
              <li className="flex items-start gap-2">
                <span className="text-f1-red">•</span>
                Factors in track characteristics
              </li>
              <li className="flex items-start gap-2">
                <span className="text-f1-red">•</span>
                Accuracy improves through season
              </li>
            </ul>
          </div>
        </div>

        {/* Prediction Display */}
        <div className="lg:col-span-2">
          {/* Selected Race Header */}
          {selectedRace && (
            <div className="card-glass mb-6 animate-fade-in">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                  <h2 className="text-xl font-bold text-white">
                    {selectedRace.race_name}
                  </h2>
                  <p className="text-gray-400 text-sm mt-1">
                    Round {selectedRace.round} • {selectedRace.date}
                  </p>
                </div>
                
                {!predictionRequested && (
                  <button
                    onClick={handleGeneratePrediction}
                    className="btn-primary"
                  >
                    Generate Prediction
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Loading */}
          {predictionsLoading && (
            <div className="py-16">
              <LoadingSpinner message="Generating prediction..." />
            </div>
          )}

          {/* Error */}
          {predictionsError && (
            <ErrorMessage message={(predictionsError as Error).message} />
          )}

          {/* Predictions */}
          {predictions && !predictionsLoading && (
            <div className="space-y-6 animate-fade-in-up">
              {/* Prediction Label */}
              <div className="flex items-center justify-between">
                <span className="chip-info">
                  Pre-Race Prediction
                </span>
                <button
                  onClick={() => navigate(`/dashboard?year=${selectedYear}&round=${selectedRound}`)}
                  className="text-f1-red text-sm hover:underline"
                >
                  View Full Dashboard →
                </button>
              </div>

              {/* Podium */}
              <PodiumCard predictions={predictions.predictions} />

              {/* Predictions Table */}
              <div className="card">
                <h3 className="text-lg font-semibold text-white mb-4">
                  Predicted Finishing Order
                </h3>
                <PredictionTable 
                  predictions={predictions.predictions}
                  showActual={false}
                />
              </div>
            </div>
          )}

          {/* Empty State */}
          {!predictionRequested && !predictions && selectedRace && (
            <div className="card-glass text-center py-16">
              <div className="text-5xl mb-4">🏁</div>
              <h3 className="text-xl font-semibold text-white mb-2">
                Ready to Predict
              </h3>
              <p className="text-gray-400 max-w-md mx-auto mb-6">
                Click "Generate Prediction" to see the predicted finishing order 
                for {selectedRace.race_name}.
              </p>
              <button
                onClick={handleGeneratePrediction}
                className="btn-primary"
              >
                Generate Prediction
              </button>
            </div>
          )}

          {/* No Race Selected */}
          {!selectedRace && !racesLoading && (
            <div className="card-glass text-center py-16">
              <div className="text-5xl mb-4">📅</div>
              <h3 className="text-xl font-semibold text-white mb-2">
                Select a Race
              </h3>
              <p className="text-gray-400">
                Choose a race from the list to generate predictions.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

