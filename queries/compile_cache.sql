-- Graph compilation: cold compile against a cache hit.
SELECT count(*) FILTER (WHERE compile_s IS NOT NULL) AS cold_runs,
       round(median(compile_s), 1) AS cold_median_s,
       count(*) FILTER (WHERE compile_cache_hit_s IS NOT NULL) AS cache_hit_runs,
       round(median(compile_cache_hit_s), 1) AS cache_hit_median_s
FROM 'tables/startup_runs.parquet';
