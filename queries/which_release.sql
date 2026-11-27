-- Which releases of each runtime load this architecture natively?
SELECT runtime, first_version, last_version, releases, in_latest
FROM 'tables/architecture_ranges.parquet'
WHERE architecture = 'Qwen3ForCausalLM';
