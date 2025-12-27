import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../lib/api';
import { RaceSelector } from '../components/RaceSelector';
import type { Race } from '../types';

export function Landing() {
  const navigate = useNavigate();
  const [selectedYear, setSelectedYear] = useState(2024);
  const [selectedRound, setSelectedRound] = useState(1);

  const { data: status } = useQuery({
    queryKey: ['status'],
    queryFn: () => apiClient.getStatus(),
  });

  const { data: races } = useQuery({
    queryKey: ['races', selectedYear],
    queryFn: () => apiClient.getRaces(selectedYear),
    enabled: selectedYear > 0,
  });

  const selectedRace = races?.find((r: Race) => r.round === selectedRound);

  const handleGeneratePrediction = () => {
    navigate(`/dashboard?year=${selectedYear}&round=${selectedRound}`);
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Hero Section */}
      <section className="flex-1 flex items-center justify-center px-4 py-16">
        <div className="max-w-4xl mx-auto text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 bg-f1-red/20 border border-f1-red/30 
                         rounded-full px-4 py-2 mb-8 animate-fade-in">
            <span className="w-2 h-2 bg-f1-red rounded-full animate-pulse" />
            <span className="text-f1-red text-sm font-medium">2026 Season Ready</span>
          </div>

          {/* Main Title */}
          <h1 className="font-racing text-4xl sm:text-5xl md:text-7xl text-white mb-6 
                        animate-fade-in-up leading-tight">
            Predict F1 Race
            <span className="block text-f1-red">Results</span>
          </h1>

          <p className="text-gray-400 text-lg sm:text-xl mb-12 max-w-2xl mx-auto 
                       animate-fade-in-up stagger-1">
            Machine learning powered predictions using 8 years of historical data. 
            Select a race and see the predicted finishing order.
          </p>

          {/* Race Selection Card */}
          <div className="card-glass max-w-xl mx-auto animate-fade-in-up stagger-2">
            <h2 className="text-lg font-semibold text-white mb-6 text-left">
              Select a Race
            </h2>

            <RaceSelector
              selectedYear={selectedYear}
              selectedRound={selectedRound}
              onYearChange={setSelectedYear}
              onRoundChange={setSelectedRound}
              availableYears={status?.available_years || [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]}
            />

            {/* Selected Race Preview */}
            {selectedRace && (
              <div className="mt-6 p-4 bg-f1-dark/50 rounded-lg border border-gray-700">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-white font-medium">{selectedRace.race_name}</div>
                    <div className="text-gray-400 text-sm">
                      Round {selectedRace.round} • {selectedRace.date}
                    </div>
                  </div>
                  <div className="text-3xl font-racing text-f1-red">
                    R{selectedRace.round}
                  </div>
                </div>
              </div>
            )}

            {/* CTA Button */}
            <button
              onClick={handleGeneratePrediction}
              disabled={!selectedRace}
              className="btn-primary w-full mt-6 text-lg py-4 animate-pulse-glow 
                        disabled:opacity-50 disabled:cursor-not-allowed disabled:animate-none"
            >
              Generate Prediction
            </button>
          </div>

          {/* Quick Stats */}
          <div className="grid grid-cols-3 gap-4 mt-12 max-w-lg mx-auto animate-fade-in-up stagger-3">
            <div className="text-center">
              <div className="text-2xl font-bold text-white">{status?.available_years?.length || 8}</div>
              <div className="text-gray-500 text-sm">Seasons</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-white">173</div>
              <div className="text-gray-500 text-sm">Races</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-f1-red">LightGBM</div>
              <div className="text-gray-500 text-sm">Model</div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="border-t border-gray-800 py-16 px-4">
        <div className="max-w-6xl mx-auto">
          <h2 className="font-racing text-2xl text-white text-center mb-12">
            How It Works
          </h2>

          <div className="grid md:grid-cols-3 gap-8">
            <FeatureCard
              icon="📊"
              title="Historical Data"
              description="Trained on 8 years of F1 data including qualifying results, driver form, and track characteristics."
              delay={0.1}
            />
            <FeatureCard
              icon="🤖"
              title="ML Ranking Model"
              description="LightGBM ranking model predicts finishing positions using pre-race data only."
              delay={0.2}
            />
            <FeatureCard
              icon="🔄"
              title="Rolling Updates"
              description="Model improves throughout the season as it learns from each completed race."
              delay={0.3}
            />
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 px-4 racing-stripes">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="font-racing text-2xl sm:text-3xl text-white mb-6">
            Ready for the 2026 Season?
          </h2>
          <p className="text-gray-400 mb-8">
            Make predictions before each race and see how accurate the model becomes.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={() => navigate('/upcoming')}
              className="btn-primary text-lg"
            >
              Upcoming Races
            </button>
            <button
              onClick={() => navigate('/dashboard?year=2024&round=24')}
              className="btn-outline text-lg"
            >
              View Latest Analysis
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}

function FeatureCard({ 
  icon, 
  title, 
  description, 
  delay 
}: { 
  icon: string; 
  title: string; 
  description: string; 
  delay: number;
}) {
  return (
    <div 
      className="card-hover text-center animate-fade-in-up"
      style={{ animationDelay: `${delay}s` }}
    >
      <div className="text-4xl mb-4">{icon}</div>
      <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
      <p className="text-gray-400 text-sm">{description}</p>
    </div>
  );
}

