import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../lib/api';
import type { Race } from '../types';

interface RaceSelectorProps {
  selectedYear: number;
  selectedRound: number;
  onYearChange: (year: number) => void;
  onRoundChange: (round: number) => void;
  availableYears?: number[];
  showUpcoming?: boolean;
}

export function RaceSelector({
  selectedYear,
  selectedRound,
  onYearChange,
  onRoundChange,
  availableYears = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
  showUpcoming = false,
}: RaceSelectorProps) {
  const { data: races, isLoading: racesLoading } = useQuery({
    queryKey: ['races', selectedYear],
    queryFn: () => apiClient.getRaces(selectedYear),
    enabled: selectedYear > 0,
  });

  const years = showUpcoming 
    ? [...availableYears, 2026] 
    : availableYears;

  const handleYearChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const year = Number(e.target.value);
    onYearChange(year);
    onRoundChange(1);
  };

  const handleRoundChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onRoundChange(Number(e.target.value));
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <label className="block text-sm font-medium text-gray-400 mb-2">
          Season
        </label>
        <select
          className="select w-full"
          value={selectedYear}
          onChange={handleYearChange}
        >
          {years.map((year) => (
            <option key={year} value={year}>
              {year} {year === 2026 ? '(Upcoming)' : 'Season'}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-400 mb-2">
          Race
        </label>
        <select
          className="select w-full"
          value={selectedRound}
          onChange={handleRoundChange}
          disabled={racesLoading || !races}
        >
          {racesLoading ? (
            <option>Loading races...</option>
          ) : races && races.length > 0 ? (
            races.map((race: Race) => (
              <option key={race.round} value={race.round}>
                R{race.round} — {race.race_name}
              </option>
            ))
          ) : (
            <option>No races available</option>
          )}
        </select>
      </div>
    </div>
  );
}

// Compact version for inline use
export function RaceSelectorCompact({
  selectedYear,
  selectedRound,
  onYearChange,
  onRoundChange,
  availableYears = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
}: RaceSelectorProps) {
  const { data: races, isLoading } = useQuery({
    queryKey: ['races', selectedYear],
    queryFn: () => apiClient.getRaces(selectedYear),
    enabled: selectedYear > 0,
  });

  return (
    <div className="flex items-center gap-3 flex-wrap">
      <select
        className="select text-sm py-2"
        value={selectedYear}
        onChange={(e) => {
          onYearChange(Number(e.target.value));
          onRoundChange(1);
        }}
      >
        {availableYears.map((year) => (
          <option key={year} value={year}>{year}</option>
        ))}
      </select>

      <select
        className="select text-sm py-2 min-w-[200px]"
        value={selectedRound}
        onChange={(e) => onRoundChange(Number(e.target.value))}
        disabled={isLoading || !races}
      >
        {isLoading ? (
          <option>Loading...</option>
        ) : races?.map((race: Race) => (
          <option key={race.round} value={race.round}>
            R{race.round} — {race.race_name}
          </option>
        ))}
      </select>
    </div>
  );
}

