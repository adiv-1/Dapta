export type Adapter = {
  adapter_id: string;
  skill_id: string;
  version: string;
  base_model: string;
  artifact_uri: string;
  created_at: string;
  metadata: Record<string, unknown>;
};

export type Condition = {
  status: string;
  model: string;
  label?: string;
  output: Record<string, unknown>;
  error?: string;
  metrics: {
    latency_ms: number | null;
    ttft_ms?: number | null;
    input_tokens?: number;
    output_tokens?: number;
    cost_usd?: string | null;
    accuracy?: number;
    schema_valid?: boolean;
    accuracy_label?: string;
  };
};

export type Run = {
  run_id: string;
  workflow: string;
  adapter_id: string;
  attachment_name?: string;
  status: string;
  results: Record<string, Condition>;
  summary: Record<string, string | null>;
  error_message?: string;
  created_at: string;
  updated_at: string;
};

export type Benchmark = {
  benchmark_id: string;
  name: string;
  created_at: string;
  view: string;
  test_documents: number;
  test_sha256: string;
  conditions: Record<string, Record<string, number | string>>;
};

export type Job = {
  job_id: string;
  workflow: string;
  skill_id?: string;
  status: string;
  created_at: string;
  updated_at: string;
  events?: Array<{ to_status: string; occurred_at: string }>;
};
