# Database seeds

The repository ignores `*.sql` by default to prevent large dumps from being committed.

For local development:
- Use the sample seed scripts in this folder if available, or create your own small seeds.
- The helper `scripts/seed_db.sh` copies `scripts/seed_app.sql` and `scripts/seed_scrapers.sql` into the running `postgres` container and applies them.
- If you need to commit seed examples, rename them with a suffix like `.sql.example` to bypass the ignore rule.