-- Architectures a runtime used to load natively and no longer does.
SELECT runtime, architecture, first_version, last_version AS last_supported
FROM 'tables/architecture_ranges.parquet'
WHERE NOT in_latest
ORDER BY runtime, last_version DESC;
