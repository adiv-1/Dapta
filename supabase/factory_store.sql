create table if not exists public.dapta_factory_runs (
  job_id text primary key,
  workflow_hash text not null,
  skill_id text,
  status text not null,
  dataset_round integer not null default 1,
  failure_code text,
  failure_message text,
  artifact_uri text,
  created_at timestamptz not null,
  updated_at timestamptz not null
);

create table if not exists public.dapta_factory_events (
  event_id uuid primary key,
  job_id text not null references public.dapta_factory_runs(job_id) on delete cascade,
  sequence bigint not null,
  from_status text,
  to_status text not null,
  occurred_at timestamptz not null,
  metadata jsonb not null default '{}'::jsonb,
  unique(job_id, sequence)
);

create table if not exists public.dapta_factory_artifacts (
  job_id text not null references public.dapta_factory_runs(job_id) on delete cascade,
  name text not null,
  sha256 text not null,
  payload jsonb not null,
  created_at timestamptz not null,
  primary key(job_id, name)
);

create index if not exists dapta_factory_events_job_time
  on public.dapta_factory_events(job_id, occurred_at);

alter table public.dapta_factory_runs enable row level security;
alter table public.dapta_factory_events enable row level security;
alter table public.dapta_factory_artifacts enable row level security;

revoke all on public.dapta_factory_runs from anon, authenticated;
revoke all on public.dapta_factory_events from anon, authenticated;
revoke all on public.dapta_factory_artifacts from anon, authenticated;

grant all on public.dapta_factory_runs to service_role;
grant all on public.dapta_factory_events to service_role;
grant all on public.dapta_factory_artifacts to service_role;
