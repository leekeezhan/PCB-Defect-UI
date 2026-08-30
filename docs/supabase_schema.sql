-- ---------------------------------------------------------------------------
-- PCB Defect Inspection System — inspection history table
--
-- Run this once in the Supabase SQL editor (Dashboard → SQL Editor → New query).
-- The interface then writes one row per inspected board when the sidebar's
-- history store is set to Supabase, and the History page reads them back.
--
-- Client: Student4-UI/core/storage.py  (class SupabaseStore)
-- ---------------------------------------------------------------------------

create table if not exists public.inspections (
    id                bigint generated always as identity primary key,

    -- When the board was inspected, in UTC. Written by the client rather than
    -- defaulted here, so a row keeps the inspecting machine's clock reading
    -- even if it is uploaded later.
    inspected_at      timestamptz  not null default now(),

    -- What was inspected: a file name, "frame_42", or a camera label.
    source            text         not null,

    -- Which page produced the row: single | batch | video | live.
    mode              text         not null,

    -- The verdict from Module 4's acceptance criteria: PASS | REVIEW | FAIL.
    verdict           text         not null,

    quality_score     real         not null default 0,
    total_defects     integer      not null default 0,
    critical_defects  integer      not null default 0,
    uncertain_defects integer      not null default 0,
    mean_confidence   real         not null default 0,
    inference_ms      real         not null default 0,

    -- The detector that produced the detections, for traceability.
    model             text         not null default '',

    image_width       integer      not null default 0,
    image_height      integer      not null default 0,

    -- Per-class counts, e.g. {"missing_hole": 2, "short": 1}. Stored as text so
    -- that adding a defect class needs no schema migration; the client
    -- serialises and parses it.
    class_counts      text         not null default '{}',

    created_at        timestamptz  not null default now()
);

-- The History page sorts by time and filters by verdict, so both are indexed.
create index if not exists idx_inspections_time
    on public.inspections (inspected_at desc);

create index if not exists idx_inspections_verdict
    on public.inspections (verdict);

create index if not exists idx_inspections_mode
    on public.inspections (mode);


-- ---------------------------------------------------------------------------
-- Row-level security
--
-- Supabase enables RLS on new tables, and with no policy attached every request
-- from the anon key is refused — the interface would report "could not reach
-- the Supabase table". The policies below open insert, select and delete to the
-- anon role, which is appropriate for a coursework prototype holding no
-- personal data.
--
-- Do NOT use the service-role key in the interface to work around this: that key
-- bypasses RLS entirely and would be shipped inside a desktop application.
-- ---------------------------------------------------------------------------

alter table public.inspections enable row level security;

drop policy if exists "inspections_insert_anon" on public.inspections;
create policy "inspections_insert_anon"
    on public.inspections for insert
    to anon, authenticated
    with check (true);

drop policy if exists "inspections_select_anon" on public.inspections;
create policy "inspections_select_anon"
    on public.inspections for select
    to anon, authenticated
    using (true);

-- Required only by the History page's "Clear history" button. Drop this policy
-- if the history should be append-only.
drop policy if exists "inspections_delete_anon" on public.inspections;
create policy "inspections_delete_anon"
    on public.inspections for delete
    to anon, authenticated
    using (true);


-- ---------------------------------------------------------------------------
-- Convenience view: production yield per day.
-- Not used by the interface, but useful for a chart in the report.
-- ---------------------------------------------------------------------------

create or replace view public.inspection_yield_daily as
select
    date_trunc('day', inspected_at)                                as day,
    count(*)                                                       as boards,
    count(*) filter (where verdict = 'PASS')                       as passed,
    count(*) filter (where verdict = 'REVIEW')                     as review,
    count(*) filter (where verdict = 'FAIL')                       as failed,
    round(100.0 * count(*) filter (where verdict = 'PASS') / count(*), 1) as yield_pct,
    round(avg(quality_score)::numeric, 1)                          as mean_quality,
    sum(total_defects)                                             as total_defects
from public.inspections
group by 1
order by 1 desc;
