# HearthMinds Database Documentation

## 1. Core Design Principles

### Error Handling ✅
- [x] Fail fast, fail hard - no error recovery
- [x] All operations must be fully destructive
- [x] Treat all data as transitory (ETL approach)
- [x] No rollback mechanisms needed

### Testing Strategy ✅
- [x] Boolean-based action validation
- [x] Pattern documented in `eng_patterns` table
- [x] Runtime logging of all boolean results
- [x] No traditional unit tests needed at this stage

The boolean validation pattern is now our first engineering pattern! Query it with:
```sql
SELECT content FROM eng_patterns 
WHERE tags @> ARRAY['testing', 'validation'];
```

### Security & Configuration ✅
- [x] Simple password scheme: passwords match usernames
- [x] Configuration via environment and CLI args
- [x] Clean separation of concerns in setup module
- [x] Service account integration
  - [x] hearthminds-service system user
  - [x] Proper database ownership
  - [x] Peer authentication setup
  - [x] Secure extension management

### Infrastructure ✅
- [x] Bare-metal Debian deployment
- [x] Proto-person database architecture
  - [x] Separate databases per AI
  - [x] Shared infrastructure in HearthMinds DB
  - [x] Vector-enabled schemas
- [x] Nuclear reset capability for clean state

## 2. Database Structure

### Hearthminds Infrastructure Database
- Owner: hearthminds-service
- Manager: hearthminds (superuser)
- Purpose: Shared infrastructure and network management
- Extensions: vector (owned by hearthminds)
- Cross-DB Access: Granted to all AI users
- Authentication: Peer auth for service account

### AI-Specific Databases
- One database per AI (e.g., logos, aletheia)
- Owner: Respective AI user
- Extensions: vector
- Cross-DB Access: hearthminds user has monitoring access

## 3. Environment Configuration

Required environment variables:
```ini
# PostgreSQL Installation
PG_VERSION=15
PG_CLUSTER_NAME=main

# Database Configuration
POSTGRES_HOST=10.0.0.1
POSTGRES_PORT=5432

# Service Account
HEARTHMINDS_SERVICE_USER=hearthminds-service  # System user for service operations

# Admin Configuration
POSTGRES_ADMIN_USER=postgres
POSTGRES_ADMIN_PASSWORD=your_superuser_password

# HearthMinds Database (Project-wide)
HEARTHMINDS_DB=hearthminds_db
HEARTHMINDS_USER=hearthminds
HEARTHMINDS_PASSWORD=your_hearthminds_password

# AI-Specific Configuration
AI_NAMES=["logos", "aletheia"]           # Array of AI names
AI_DB_SUFFIX="_db"                       # Optional suffix for databases
AI_DEFAULT_EXTENSIONS=["vector"]         # Array of extensions to install

# Per-AI Credentials (Password matches username for simplicity)
${AI_NAME}_PASSWORD=${AI_NAME}           # e.g., logos_password=logos
```

## 4. Database Migrations

This project uses Alembic for schema migrations. In development, the app may still create
extensions/tables for convenience, but production and team workflows should rely on Alembic.

### Setup
- Ensure your virtualenv is active and dev deps are installed:
```bash
make setup
```

### Common Commands
- Create a new revision:
```bash
make migrate-rev m="add new table"
```
- Upgrade to latest:
```bash
make migrate-up
```
- Downgrade one step:
```bash
make migrate-down
```

### Migration History

#### M1: Baseline (0001)
- Creates `memory_shards` table
- Ensures `vector` extension exists
- Configurable embedding dimension via `EMBEDDING_DIM`

#### M2: IVFFlat Index (0003)
- Creates IVFFlat index on `memory_shards.embedding`
- Configurable via:
  - `PGVECTOR_ENABLE_IVFFLAT` (default: true)
  - `PGVECTOR_IVFFLAT_LISTS` (default: 100)

#### M3: Content Metadata (0004, 0005)
- Adds `source` and `metadata` columns to `memory_shards`
- Creates `raw_conversations` table for request/response logging

#### M4: Engineering Patterns (0006)
- Adds `eng_patterns` table for RAG-based pattern management
- Supports semantic search via pgvector embeddings
- Used for AI-assisted development pattern retrieval

## 5. Phase 2 Roadmap

1. Pattern Documentation
   - [ ] Add error handling patterns
   - [ ] Document database access patterns
   - [ ] API integration patterns

2. Schema Evolution
   - [ ] Vector similarity search optimization
   - [ ] Memory shard compression strategies
   - [ ] Bond history analytics

3. Monitoring (Future)
   - [ ] Consider operation timing metrics
   - [ ] Pattern usage analytics
   - [ ] Schema version tracking

## 6. Troubleshooting

### Existing Tables, No Alembic State
If you see `psycopg.errors.DuplicateTable` errors during `alembic upgrade`:

1. Confirm tables exist:
```bash
docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\\dt"
```

2. Stamp baseline:
```bash
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/aletheia \
  ./.venv/bin/alembic stamp 0001
```

3. Upgrade to head:
```bash
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/aletheia \
  ./.venv/bin/alembic upgrade head
```

### Notes
- Do not commit real secrets
- Keep migrations deterministic and reversible
- Always `ANALYZE` after big ingests or index builds
- Prefer migrations in production over runtime table creation