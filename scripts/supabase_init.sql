-- GeoFlare Supabase bootstrap
-- Run this once in the Supabase SQL editor (Dashboard -> SQL Editor -> New query).
-- Everything else (user_fire_detections table, geom index) is created automatically
-- by src/database.py:init_db() on first request, as long as PostGIS is available.

CREATE EXTENSION IF NOT EXISTS postgis;

-- Sanity checks
SELECT postgis_version();
SELECT extname FROM pg_extension WHERE extname = 'postgis';