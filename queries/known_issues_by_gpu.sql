-- Known startup failures per GPU family.
SELECT family, count(*) AS issues
FROM (SELECT unnest(gpu_families) AS family FROM 'tables/known_issues.parquet')
GROUP BY family ORDER BY issues DESC;
