# Integration Phase 1: Local Deployment & SQLite Migration

## Overview
This phase focuses on deploying the containerized API stack to local hardware (logos @ 10.0.0.1) and migrating existing SQLite conversation history into the PostgreSQL database schema.

## Goals
1. Deploy API containers to on-prem hardware
2. Migrate SQLite conversation history to PostgreSQL
3. Validate data integrity and API functionality
4. Establish baseline for future enhancements

## Prerequisites
- PostgreSQL 15 running on logos (10.0.0.1:5432) ✅ CONFIRMED
- Docker installed on logos
- WireGuard VPN connectivity ✅ CONFIRMED
- SQLite conversation history available for export
- Access credentials configured

## Architecture

```
Local Development (penguin)          On-Prem Server (logos @ 10.0.0.1)
├── Project repository               ├── PostgreSQL 15 (hearthminds DB)
├── .env configuration                ├── Docker containers:
└── VPN connection                    │   ├── aletheia-api (FastAPI)
                                      │   └── openwebui (optional)
                                      └── Persistent volumes
```

## Phase 1 Tasks

### 1. Container Preparation
**Objective**: Prepare Docker images and deployment configuration

- [ ] Review `docker-compose.yml` for production readiness
- [ ] Update database connection strings for remote PostgreSQL
- [ ] Configure environment variables for on-prem deployment
- [ ] Test container builds locally
- [ ] Document any custom configuration needed

**Files to review**:
- `docker-compose.yml`
- `docker-compose.override.yml`
- `.env` (update for production)
- `Dockerfile`

### 2. Database Schema Setup
**Objective**: Ensure PostgreSQL schema is current and ready

- [ ] Verify Alembic migrations are up to date
- [ ] Connect to logos PostgreSQL from development machine
- [ ] Run migrations against production database
- [ ] Verify pgvector extension is installed
- [ ] Validate all tables/indexes exist

**Commands**:
```bash
# Test connection (from penguin)
psql -h 10.0.0.1 -U chapinad -d hearthminds -c "SELECT version();"

# Run migrations
alembic upgrade head

# Verify schema
psql -h 10.0.0.1 -U chapinad -d hearthminds -c "\dt"
```

### 3. SQLite Export & Migration
**Objective**: Transfer conversation history from SQLite to PostgreSQL

#### 3.1 Export from SQLite
- [x] Identify SQLite database location and structure
- [x] Document table schema (columns, types, relationships)
- [x] Export data to CSV or JSON format
- [x] Validate export completeness (row counts, data integrity)

**Export format needed**:
```csv
id,conversation_id,author,role,content,timestamp,parent_id
```

Or JSON:
```json
{
  "messages": [
    {
      "id": "uuid",
      "conversation_id": "uuid", 
      "author": "user|assistant",
      "role": "user|assistant",
      "content": "message text",
      "timestamp": "ISO 8601",
      "parent_id": "uuid or null"
    }
  ]
}
```

#### 3.2 Create Migration Script
- [x] Build Python script to import CSV/JSON into PostgreSQL
- [x] Handle UUID generation/mapping
- [x] Create user profiles (Aaron, Aletheia identity)
- [x] Create bond_history entries
- [x] Import messages into raw_conversations table
- [ ] Generate embeddings (deferred to Phase 2)
- [ ] Calculate metrics (deferred to Phase 2)

**Script structure**:
```python
# app/db/scripts/import_sqlite_history.py
- Load CSV/JSON export
- Connect to PostgreSQL (10.0.0.1)
- Create/verify user profiles
- Create/verify identity profiles (Aletheia)
- Create bond_history entries
- Batch import messages
- Generate embeddings (if API key available)
- Validate import (row counts, relationships)
```

#### 3.3 Validation
- [x] Verify row counts match export (13,513 imported from 15,108 total, ~99% of valid messages)
- [x] Check user_id consistency (all set to Aaron UUID)
- [x] Validate parent_id relationships (894 orphaned due to skipped empty messages)
- [ ] Test semantic search queries (deferred until embeddings generated)
- [x] Spot-check content integrity

### 4. Container Deployment
**Objective**: Deploy API stack to logos server

#### 4.1 Pre-deployment
- [ ] Copy docker-compose files to logos
- [ ] Copy .env with production settings
- [ ] Create necessary directories/volumes on logos
- [ ] Test network connectivity (logos can reach itself on localhost)

#### 4.2 Deployment
```bash
# On logos server
cd ~/openai_pgvector_api
docker-compose up -d

# Verify containers running
docker-compose ps

# Check logs
docker-compose logs -f aletheia-api
```

#### 4.3 Smoke Tests
- [ ] API health check: `curl http://10.0.0.1:8000/health`
- [ ] Test embeddings endpoint
- [ ] Test chat endpoint (with/without memory retrieval)
- [ ] Verify database connectivity from container
- [ ] Test VPN access from development machine

### 5. API Validation
**Objective**: Confirm all endpoints work with migrated data

- [ ] `/v1/chat/completions` - basic chat
- [ ] `/v1/embeddings` - embedding generation
- [ ] `/api/memories` - retrieve user memories
- [ ] `/api/memories/search` - semantic search
- [ ] User-scoped memory retrieval (Aaron's conversations)
- [ ] Performance benchmarks (response times, embedding latency)

## Success Criteria
- ✅ PostgreSQL schema deployed and validated
- ✅ All SQLite conversation history migrated (10k+ messages)
- ✅ API containers running on logos
- ✅ Endpoints accessible via VPN (http://10.0.0.1:8000)
- ✅ Memory retrieval working with user scoping
- ✅ Zero data loss from SQLite → PostgreSQL
- ✅ No hardcoded secrets in deployed configs

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Data loss during migration | High | Export validation, dry-run imports, backups |
| Connection issues (VPN/DB) | Medium | Test connectivity early, document firewall rules |
| Embedding generation slow | Low | Defer embeddings to Phase 2 if needed |
| Docker resource constraints | Medium | Monitor logos resources, adjust container limits |
| Schema mismatches | High | Run migrations first, validate before import |

## Timeline Estimate
- Container prep & testing: 2-4 hours
- Database schema setup: 1-2 hours
- SQLite export & migration script: 4-6 hours
- Migration execution & validation: 2-3 hours
- Container deployment: 1-2 hours
- API validation & smoke tests: 2-3 hours

**Total**: 12-20 hours (1-2 days of focused work)

## Next Steps (Phase 2)
- Backfill embeddings for imported messages
- Implement Google SSO authentication
- Add monitoring/logging (Grafana, Prometheus)
- Performance tuning (pgvector indexes, caching)
- OpenWebUI integration and customization

## Notes
- Keep SQLite database as backup until Phase 1 fully validated
- Document any schema changes needed for migration
- Track edge cases encountered during import
- Note any data cleaning/normalization required

## Open Questions
1. What is the exact schema of the SQLite database?
2. Are there existing UUIDs, or do we generate them?
3. Should embeddings be generated during import, or deferred?
4. What's the expected message volume (10k? 50k? 100k+)?
5. Are there multiple users in SQLite, or just Aaron ↔ Aletheia?
