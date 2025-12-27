import type { StatusResponse, Race, PredictionResponse, UpdateResponse } from '../types';

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
}

export const apiClient = new ApiClient(API_BASE_URL);

