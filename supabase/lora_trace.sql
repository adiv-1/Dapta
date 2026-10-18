-- Full LoRA-agent trace. Run this once in the Supabase SQL editor.
-- The secret key writes these rows. No public policy is added.

create table if not exists lora_runs (
  run_id uuid primary key,
  task text not null,
  skill_id text not null,
  device text,
  status text not null,
  artifact_uri text,
  created_at timestamptz not null default now()
);

create table if not exists lora_events (
  event_id uuid primary key default gen_random_uuid(),
  run_id uuid not null references lora_runs(run_id),
  stage text not null,
  agent text not null,
  prompt text not null,
  thought jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists lora_events_run_time
  on lora_events(run_id, created_at);

alter table lora_runs enable row level security;
alter table lora_events enable row level security;
