import type { StatusResponse, Race, PredictionResponse, UpdateResponse, ChatResponse } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `API error: ${response.status}`);
    }

    return response.json();
  }

  async getStatus(): Promise<StatusResponse> {
    return this.request<StatusResponse>('/status');
  }

  async getRaces(year: number): Promise<Race[]> {
    return this.request<Race[]>(`/races/${year}`);
  }

  async getPredictions(year: number, round: number): Promise<PredictionResponse> {
    return this.request<PredictionResponse>(`/predict?year=${year}&round=${round}`);
  }

  async updateAfterRace(year: number, round: number, retrain: boolean = true): Promise<UpdateResponse> {
    return this.request<UpdateResponse>(
      `/update?year=${year}&round=${round}&retrain=${retrain}`,
      { method: 'POST' }
    );
  }

  async refreshData(years: string = '2018-2025'): Promise<{ status: string; message: string }> {
    return this.request<{ status: string; message: string }>(
      `/refresh-data?years=${years}`,
      { method: 'POST' }
    );
  }

  async askChatbot(question: string, useLlm: boolean = true): Promise<ChatResponse> {
    return this.request<ChatResponse>(
      `/chat?question=${encodeURIComponent(question)}&use_llm=${useLlm}`,
      { method: 'POST' }
    );
  }

  async prepareRace(year: number, race: string, buildFeatures: boolean = true): Promise<{
    ok: boolean;
    year: number;
    race: string;
    data_fetched: boolean;
    laps_added?: number;
    results_added?: number;
    weather_added?: number;
    features_built?: boolean;
    features_path?: string;
    features_error?: string;
  }> {
    return this.request<{
      ok: boolean;
      year: number;
      race: string;
      data_fetched: boolean;
      laps_added?: number;
      results_added?: number;
      weather_added?: number;
      features_built?: boolean;
      features_path?: string;
      features_error?: string;
    }>(
      `/prepare-race?year=${year}&race=${encodeURIComponent(race)}&build_features=${buildFeatures}`,
      { method: 'POST' }
    );
  }
}

export const apiClient = new ApiClient(API_BASE_URL);

