-- ---------------------------------------------------------------------------
-- PCB Defect Inspection System — stored inspection images
--
-- Run this in the Supabase SQL editor AFTER docs/supabase_schema.sql.
-- It adds the columns that hold each board's pictures, and gives the anon key
-- permission to write into the three buckets the team already uses.
--
-- What gets stored: four JPEGs per inspected board, roughly 30-200 KB each —
-- the operator's input, Module 1's output, Module 2's output and the annotated
-- detection result. Videos are NOT stored: one 12-second 720p clip is ~35 MB
-- and would fill the free plan's 1 GB after about 29 recordings.
--
--     stage         bucket           written by
--     ------------  ---------------  ------------------------------------------
--     original      pcb-uploads      what the operator supplied
--     preprocessed  pcb-processed    Module 1
--     aligned       pcb-processed    Module 2
--     annotated     pcb-annotated    Module 3's detections drawn by Module 4
--
-- Client: core/storage.py  (SupabaseStore.upload_images, DEFAULT_BUCKETS)
-- ---------------------------------------------------------------------------

-- ---------------------------------------------------------------------------
-- 1. Columns holding the public URL of each stage's picture.
--
-- URLs rather than paths, so a row is self-contained: the History page, a CSV
-- export and the report can all use it without knowing the project or bucket.
-- Empty string means "not stored" — image upload is optional and can be turned
-- off in the sidebar, and the SQLite store never fills these in.
-- ---------------------------------------------------------------------------
alter table public.inspections
    add column if not exists image_original     text not null default '',
    add column if not exists image_preprocessed text not null default '',
    add column if not exists image_aligned      text not null default '',
    add column if not exists image_annotated    text not null default '';


-- ---------------------------------------------------------------------------
-- 2. The buckets.
--
-- Created only if they are missing, so an existing bucket keeps the size limit
-- and MIME restrictions it was set up with. They are public: the History page
-- shows the pictures as ordinary <img> sources, which needs no signed URL and
-- no extra round trip. Board photographs carry nothing personal.
-- ---------------------------------------------------------------------------
insert into storage.buckets (id, name, public)
values ('pcb-uploads',   'pcb-uploads',   true),
       ('pcb-processed', 'pcb-processed', true),
       ('pcb-annotated', 'pcb-annotated', true)
on conflict (id) do nothing;


-- ---------------------------------------------------------------------------
-- 3. Row-level security on the objects.
--
-- THIS IS THE PART THAT MATTERS. Supabase enables RLS on storage.objects, so a
-- bucket showing 0 policies in the dashboard refuses every request from the
-- anon key — the upload fails and the interface reports it, having stored the
-- inspection result but no picture.
--
-- One policy per bucket per operation, each with a plain `bucket_id = '...'`,
-- rather than one policy covering all three with `bucket_id in (...)`. Both
-- grant the same access, but the dashboard's per-bucket policy count is derived
-- by reading the policy definition, and it does not attribute an IN-list to
-- every bucket named in it — so the combined form leaves two of the three
-- buckets still showing 0 and no way to confirm the grant from the screen.
--
-- Each policy is scoped to one bucket, so nothing here can reach any other
-- bucket in the project. Insert and select are what the interface needs;
-- delete is only for the History page's "Clear history" button.
--
-- These are additive: a teammate's existing policy on the same bucket is left
-- alone, because the names below are specific to this module.
-- ---------------------------------------------------------------------------

-- pcb-uploads — the operator's own input ------------------------------------
drop policy if exists "pcb_uploads_insert_anon" on storage.objects;
create policy "pcb_uploads_insert_anon"
    on storage.objects for insert
    to anon, authenticated
    with check (bucket_id = 'pcb-uploads');

drop policy if exists "pcb_uploads_select_anon" on storage.objects;
create policy "pcb_uploads_select_anon"
    on storage.objects for select
    to anon, authenticated
    using (bucket_id = 'pcb-uploads');

drop policy if exists "pcb_uploads_delete_anon" on storage.objects;
create policy "pcb_uploads_delete_anon"
    on storage.objects for delete
    to anon, authenticated
    using (bucket_id = 'pcb-uploads');

-- pcb-processed — Module 1 and Module 2 output ------------------------------
drop policy if exists "pcb_processed_insert_anon" on storage.objects;
create policy "pcb_processed_insert_anon"
    on storage.objects for insert
    to anon, authenticated
    with check (bucket_id = 'pcb-processed');

drop policy if exists "pcb_processed_select_anon" on storage.objects;
create policy "pcb_processed_select_anon"
    on storage.objects for select
    to anon, authenticated
    using (bucket_id = 'pcb-processed');

drop policy if exists "pcb_processed_delete_anon" on storage.objects;
create policy "pcb_processed_delete_anon"
    on storage.objects for delete
    to anon, authenticated
    using (bucket_id = 'pcb-processed');

-- pcb-annotated — the detection result --------------------------------------
drop policy if exists "pcb_annotated_insert_anon" on storage.objects;
create policy "pcb_annotated_insert_anon"
    on storage.objects for insert
    to anon, authenticated
    with check (bucket_id = 'pcb-annotated');

drop policy if exists "pcb_annotated_select_anon" on storage.objects;
create policy "pcb_annotated_select_anon"
    on storage.objects for select
    to anon, authenticated
    using (bucket_id = 'pcb-annotated');

drop policy if exists "pcb_annotated_delete_anon" on storage.objects;
create policy "pcb_annotated_delete_anon"
    on storage.objects for delete
    to anon, authenticated
    using (bucket_id = 'pcb-annotated');


-- ---------------------------------------------------------------------------
-- 3b. Remove the earlier combined policies, if a previous run of this file
--     created them. They are superseded by the nine above.
-- ---------------------------------------------------------------------------
drop policy if exists "pcb_images_insert_anon" on storage.objects;
drop policy if exists "pcb_images_select_anon" on storage.objects;
drop policy if exists "pcb_images_delete_anon" on storage.objects;


-- ---------------------------------------------------------------------------
-- 4. Check it worked.
--
-- Expect nine rows. The Storage → Buckets screen should then show 3 policies
-- against pcb-processed and pcb-annotated, and three more than it had against
-- pcb-uploads.
-- ---------------------------------------------------------------------------
select
    policyname,
    cmd
from pg_policies
where schemaname = 'storage'
  and tablename = 'objects'
  and (policyname like 'pcb_uploads_%'
       or policyname like 'pcb_processed_%'
       or policyname like 'pcb_annotated_%')
order by policyname;
