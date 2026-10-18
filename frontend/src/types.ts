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
  failure_code?: string;
  failure_message?: string;
  cancellation?: { cleanup_complete?: boolean; cleanup_error?: string; message?: string };
  dataset_round?: number;
  route?: RouteDecision;
  progress?: FactoryProgress;
  validation?: Record<string, unknown>;
  deterministic_evaluation?: FactoryEvaluation;
  publish_result?: {
    artifact_uri: string;
    adapter: Adapter;
    benchmark_status?: string;
    suggested_repairs?: string[];
  };
  training?: Record<string, unknown>;
  usage?: {
    total_cost_usd?: string;
    model_cost_usd?: string;
    compute_cost_usd?: string;
    conditions?: Array<{
      condition: string;
      cost_usd?: string;
      duration_ms?: number;
      request_count?: number;
    }>;
  };
  devin_session?: {
    session_id: string;
    url?: string;
    status: string;
    pull_request_url?: string;
    acus_consumed?: number;
    status_detail?: string;
    on_demand_usage?: number;
  };
  events?: Array<{
    sequence: number;
    from_status?: string;
    to_status: string;
    occurred_at: string;
    metadata?: Record<string, unknown>;
  }>;
  stage_attempts?: Array<{
    stage: string;
    round: number;
    status: string;
    claimed_at: string;
    completed_at?: string;
  }>;
  artifacts?: Array<{ name: string; sha256: string; created_at: string }>;
  agent_outputs?: AgentOutput[];
};

export type AgentOutput = {
  stage: string;
  agent: string;
  summary: string;
  output: unknown;
  source: string;
};

export type RouteDecision = {
  decision: string;
  adapter_id: string | null;
  confidence: number;
  reason: string;
  workflow_fingerprint?: string;
  source?: string;
  requires_confirmation?: boolean;
};

export type FactoryProgress = {
  stage?: string;
  message?: string;
  current?: number;
  total?: number;
  percent?: number;
  eta_seconds?: number;
  loss?: number;
  gpu_utilization_percent?: number;
  gpu_memory_used_gb?: number;
  cost_usd?: string | number;
};

export type FactoryEvaluation = {
  passed?: boolean;
  repairable?: boolean;
  benchmark_status?: string;
  evaluation_verified?: boolean;
  failure_reason?: string;
  suggested_repairs?: string[];
  dev_feedback?: { instruction?: string };
  test_cases?: number;
  conditions?: Record<string, {
    field_f1?: number;
    exact_form_rate?: number;
    schema_pass_rate?: number;
    mean_latency_seconds?: number;
  }>;
};
