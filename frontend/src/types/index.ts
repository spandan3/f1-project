export interface StatusResponse {
  status: string;
  model_exists: boolean;
  features_exist: boolean;
  available_years: number[];
}

export interface Race {
  round: number;
  race_name: string;
  date: string;
}

export interface PredictionRow {
  pred_pos: number;
  driver: string;
  constructor: string;
  grid_pos: number;
  pred_score: number;
  actual_pos?: number;
  finish_pos?: number;
}

export interface Metrics {
  ndcg_3: number;
  ndcg_10: number;
  top3_hit: number;
  spearman_rho: number;
}

export interface PredictionResponse {
  year: number;
  round: number;
  race_name: string;
  predictions: PredictionRow[];
  metrics?: Metrics;
}

export interface UpdateResponse {
  status: string;
  message: string;
  year: number;
  round: number;
  race_name: string;
  metrics?: Metrics;
}

// Extended types for UI
export interface RaceSelection {
  year: number;
  round: number;
  raceName?: string;
}

export interface ComparisonRow extends PredictionRow {
  delta: number;
  isPodiumHit: boolean;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}
