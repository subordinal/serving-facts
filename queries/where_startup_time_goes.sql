-- Median share of startup time spent in each phase, over every run where the engine came up.
SELECT count(*) AS runs,
       round(100 * median(process_start_s / total_s)) AS process_start_pct,
       round(100 * median(engine_start_s / total_s)) AS engine_start_pct,
       round(100 * median(model_load_s / total_s)) AS model_load_pct,
       round(100 * median(engine_init_s / total_s)) AS engine_init_pct
FROM 'tables/startup_runs.parquet'
WHERE engine_ready AND total_s > 0;
