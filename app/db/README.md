# Database Management

This directory contains database initialization, schema management, and infrastructure setup for the Hearthminds PostgreSQL ecosystem.

## Architecture Overview

```
db/
├── schema/          # SQL schema definitions
│   ├── hm_schema.sql   # HearthMinds shared infrastructure
│   └── pp_schema.sql   # Proto-person specific tables
├── setup/           # Database setup and management
│   └── database.py     # Nuclear reset and validation
├── models/          # SQLAlchemy models
└── utils/           # Shared utilities

```

## Key Features

- **Nuclear Reset Capability**: Full database reconstruction with validation
- **Boolean-Based Testing**: All operations follow validate-and-verify pattern
- **Idempotent Operations**: Safe to run multiple times
- **Engineering Patterns**: Self-documenting patterns stored in `eng_patterns` table

## Usage

To perform a nuclear reset (recreate all databases):

```bash
# From project root
PYTHONPATH=/path/to/project ./nuke_database.py <target_host>
```

The script will:
1. Stop PostgreSQL service
2. Remove all databases and users
3. Reinstall PostgreSQL
4. Create databases and users
5. Deploy schemas with validation
6. Verify all operations

## Design Philosophy

We follow a strict boolean validation pattern for all operations:
1. Perform action
2. Validate result independently
3. Log outcome
4. Return boolean success

This pattern is documented in our `eng_patterns` table - check the first entry!
2. Create/reset databases and users
3. Enable required extensions
4. Apply all Alembic migrations

⚠️ Script will fail if VPN connection is not established
```

## Database Structure

- **hearthminds**: Shared infrastructure database
  - Owner: hearthminds user (superuser)
  - Contains: Engineering patterns and other shared resources
  - Grants: Read access to AI users
  - Extensions: pgvector

- **logos**: Logos's persistent memory database
  - Owner: logos user
  - Contains: Memory shards and conversation history
  - Extensions: pgvector

- **aletheia**: Aletheia's persistent memory database
  - Owner: aletheia user
  - Contains: Memory shards and conversation history
  - Extensions: pgvector

## Credentials

Default credentials are defined in the project's `.env` file:

  database/username
- Global admin: hearthminds/hearthminds
- Logos DB: logos/logos
- Aletheia DB: aletheia/aletheia

⚠️ **Warning**: The setup scripts are destructive! They will drop and recreate databases. 
Ensure you have backups before running them on any system with existing data.