"""Run every quality check against the published Parquet tables. Exits 1 if any check finds a violation."""
import sys

import duckdb

from .run import BUILD, CHECKS, TABLES


def main():
    db = duckdb.connect()
    for table in BUILD:
        db.execute(f"CREATE VIEW {table} AS SELECT * FROM '{TABLES / (table + '.parquet')}'")
    failed = 0
    for name, sql in CHECKS.items():
        violations = db.execute(sql).fetchone()[0]
        failed += bool(violations)
        print(f"{'FAIL' if violations else 'ok  '}  {name}" + (f"  ({violations} rows)" if violations else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
