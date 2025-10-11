# HearthMinds Database Design Document

This document captures our core design principles and patterns. For implementation details, see our `eng_patterns` table.

## Core Design Principles

### 1. Error Handling ✅
- [x] Fail fast, fail hard - no error recovery
- [x] All operations must be fully destructive
- [x] Treat all data as transitory (ETL approach)
- [x] No rollback mechanisms needed

### 2. Testing Strategy ✅
- [x] Boolean-based action validation
- [x] Pattern documented in `eng_patterns` table
- [x] Runtime logging of all boolean results
- [x] No traditional unit tests needed at this stage

The boolean validation pattern is now our first engineering pattern! Query it with:
```sql
SELECT content FROM eng_patterns 
WHERE tags @> ARRAY['testing', 'validation'];
```

### 3. Security & Configuration ✅
- [x] Simple password scheme: passwords match usernames
- [x] Configuration via environment and CLI args
- [x] Clean separation of concerns in setup module

### 4. Infrastructure ✅
- [x] Bare-metal Debian deployment
- [x] Proto-person database architecture
  - [x] Separate databases per AI
  - [x] Shared infrastructure in HearthMinds DB
  - [x] Vector-enabled schemas
- [x] Nuclear reset capability for clean state

### 5. Database Design ✅
- [x] Fully destructive operations for idempotency
- [x] pgvector extension management
- [x] Clear schema separation:
  - `hm_schema.sql`: Shared infrastructure (eng_patterns)
  - `pp_schema.sql`: Proto-person specific tables

## Phase 2 Roadmap

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

## Enhanced Design Plan

### 1. Environment Configuration

Required environment variables for the multi-database architecture:
```ini
# PostgreSQL Installation
PG_VERSION=15
PG_CLUSTER_NAME=main

# Database Configuration
POSTGRES_HOST=10.0.0.1
POSTGRES_PORT=5432

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

### 2. Database Structure

Based on setup_databases.sql, the database architecture consists of:

1. Hearthminds Infrastructure Database
   - Owner: hearthminds (superuser)
   - Purpose: Shared infrastructure and network management
   - Extensions: vector
   - Cross-DB Access: Granted to all AI users

2. AI-Specific Databases
   - One database per AI (e.g., logos, aletheia)
   - Owner: Respective AI user
   - Extensions: vector
   - Cross-DB Access: hearthminds user has monitoring access

### 3. Project Structure Refinement

Current files to maintain/refactor:
```
app/
  db/
    setup_databases.py       -> Split into multiple modules
    01_install_pgvector.sh  -> Merge functionality into Python
```

Proposed new structure:
```
app/
  db/
    setup/
      __init__.py
      remote.py          # SSH and command execution
      package.py         # Package management
      cluster.py        # PostgreSQL cluster management
      config.py         # Configuration management
      extension.py      # Extension installation (pgvector)
    models/
      settings.py       # Configuration dataclasses
      commands.py       # Command templates
    schema/
      pp_schema.sql        # individual proto-person schema
      hm_schema.sql     # hearthminds schema (project, network, etc)
    utils/
      ssh.py           # Enhanced SSH utilities
      validation.py    # Input validation
      retry.py         # Retry mechanisms
```

### 4. Implementation Notes

Core operations sequence:
1. [Optional] Nuclear reset:
   - Stop PostgreSQL service
   - Remove PostgreSQL packages
   - Purge configuration and data
2. Install required packages
3. Create databases (destructive)
4. Deploy schemas from SQL files:
   - `pp_schema.sql` -> AI-specific databases
   - `hm_schema.sql` -> HearthMinds database

```python
class DatabaseSetup:
    """Main orchestrator for database setup"""
    def __init__(self, host: str, nuclear: bool = False):
        self.host = host
        self.nuclear = nuclear
        self.results = []
        
    def setup(self) -> bool:
        """Runs full setup and returns overall success"""
        results = []
        
        if self.nuclear:
            results.extend([
                self.stop_postgres(),       # bool: service stopped
                self.remove_postgres(),     # bool: packages removed
                self.purge_data()          # bool: configs/data purged
            ])
            
        results.extend([
            self.install_packages(),    # bool: packages installed
            self.create_databases(),    # bool: databases exist
            self.deploy_schemas()       # bool: schemas loaded
        ])
        
        self.log_results(results)
        return all(results)
```

Note: Further validation (test writes, restores from backup) will be handled in separate phases.
    for attempt in range(max_attempts):
        try:
            return await operation()
        except PostgresError as e:
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(5)
```

### 8. Next Steps

1. Immediate TODOs:
   - Fix cluster creation and configuration issues
   - Extract configuration templates
   - Add proper error handling for each step

2. Medium-term improvements:
   - Implement proper configuration classes
   - Add structured logging
   - Create monitoring tools

3. Long-term goals:
   - Full test coverage
   - Automated rollback capabilities
   - Backup/restore support

The plan preserves working components from both implementations while addressing current issues and providing a path to a more robust solution.