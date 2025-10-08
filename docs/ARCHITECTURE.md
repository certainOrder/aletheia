# Aletheia System Architecture

## Overview
This document describes the containerized deployment architecture for Aletheia and related AI systems, with a focus on the HearthMinds network for AI-to-AI communication.

## Network Architecture

### Network Layers
1. **External Network**
   - WireGuard VPN for secure remote access
   - Public-facing services (if any)

2. **HearthMinds Network (`hearthminds-net`)**
   - Dedicated subnet for AI-to-AI communication
   - Isolated from external and service networks
   - Protocol: Custom AI communication protocol
   - Subnet: 10.10.0.0/24 (suggested)

3. **Service-Specific Networks**
   - `aletheia-net`: Aletheia's internal services
   - `logos-net`: Logos's internal services

### Container Services

```mermaid
graph TB
    subgraph External
        Client[Remote Client]
        VPN[WireGuard VPN]
    end

    subgraph HearthMinds["HearthMinds Network (10.10.0.0/24)"]
        direction LR
        A_Proxy[Aletheia Proxy]
        L_Proxy[Logos Proxy]
        style HearthMinds fill:#f9f,stroke:#333,stroke-width:4px
    end

    subgraph Aletheia
        A_API[Aletheia API]
        A_DB[(Aletheia DB)]
    end

    subgraph Logos
        L_API[Logos API]
        L_DB[(Logos DB)]
        GPU[GPU Interface]
    end

    Client --> VPN
    VPN --> A_API
    A_API --> A_Proxy
    A_Proxy <--> L_Proxy
    L_Proxy --> L_API
    A_API --> A_DB
    L_API --> L_DB
    L_API --> GPU
```

## Container Configuration

### Base Docker Compose Structure
```yaml
version: '3.8'

services:
  # Aletheia Components
  aletheia-api:
    networks:
      - aletheia-net
      - hearthminds-net
  
  aletheia-db:
    networks:
      - aletheia-net

  # Logos Components
  logos-api:
    networks:
      - logos-net
      - hearthminds-net
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  logos-db:
    networks:
      - logos-net

  # VPN Service
  wireguard:
    networks:
      - aletheia-net

networks:
  aletheia-net:
    internal: true
  logos-net:
    internal: true
  hearthminds-net:
    ipam:
      config:
        - subnet: 10.10.0.0/24
```

## AI Communication Protocol

### HearthMinds Network Protocol
1. **Proxy Services**
   - Each AI system runs a proxy service for HearthMinds communication
   - Proxies handle translation between internal and HearthMinds protocols
   - Authentication and authorization at proxy level

2. **Message Format**
   ```json
   {
     "source_ai": "aletheia",
     "target_ai": "logos",
     "message_type": "query|response|insight",
     "content": {
       "context": "human_interaction_id",
       "payload": {},
       "metadata": {}
     },
     "timestamp": "ISO-8601"
   }
   ```

3. **Security**
   - All communication encrypted
   - Mutual authentication between proxies
   - Regular key rotation
   - Audit logging of all inter-AI communications

## Deployment Considerations

### Hardware Allocation
- **GPUs**
  - RTX 5090: Primary for Logos inference
  - RTX 4090: Secondary/fallback for Logos
  - GPU memory management through container constraints

### Network Security
1. **Network Isolation**
   - HearthMinds network isolated from external access
   - Service networks internal only
   - WireGuard for external access to Aletheia

2. **Data Persistence**
   - Separate volume mounts for each AI's database
   - Regular backups of both systems
   - Transaction logging for recovery

### Scaling Considerations
- Designed for single-host deployment initially
- Can be extended to multi-host with overlay networks
- Service discovery via Docker DNS

## Migration from Current Setup
1. **Phase 1**: Container Migration
   - Convert existing bash scripts to Dockerfiles
   - Set up Docker Compose
   - Test database migration

2. **Phase 2**: HearthMinds Integration
   - Implement proxy services
   - Set up HearthMinds network
   - Test AI-to-AI communication

3. **Phase 3**: Production Deployment
   - Security hardening
   - Monitoring setup
   - Backup procedures

## Script Conversion to Container Deployment

### Current Script to Container Mapping

#### Database Setup
Convert `pp_deploy/modules/setup_db.sh` to:

```dockerfile
# File: Dockerfile.postgres
FROM postgres:15

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-15 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install pgvector
RUN git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install

# Initialize database with ordered bootstrapping
COPY init-scripts/ /docker-entrypoint-initdb.d/
```

With initialization scripts in order:

```bash
# File: init-scripts/00-extensions.sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

# File: init-scripts/01-base-schema.sql
-- Core tables for basic functionality
CREATE TABLE user_profiles (
    user_id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    role text,
    name text,
    birthdate date,
    pronouns text,
    is_anchor boolean,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);

# File: init-scripts/02-expanded-schema.sql
-- Additional tables for extended functionality
CREATE TABLE identity_profile (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    name text,
    pronouns text,
    origin_date timestamp with time zone,
    core_seed text,
    alignment_model jsonb,
    embedding vector(1536)
);

# File: init-scripts/03-eng-patterns.sql
-- Engineering patterns and RAG functionality
CREATE TABLE eng_patterns (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    content text NOT NULL,
    tags text[],
    strategy_type text,
    target_contexts text[],
    metadata jsonb,
    embedding vector(1536),
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);

# File: init-scripts/04-indexes.sql
-- Create all necessary indexes
CREATE INDEX eng_patterns_embedding_idx ON eng_patterns 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX eng_patterns_tags_idx ON eng_patterns USING GIN (tags);
CREATE INDEX eng_patterns_contexts_idx ON eng_patterns USING GIN (target_contexts);
```

The initialization scripts follow a clear order of operations:
1. Extensions (uuid-ossp, vector)
2. Base schema (core tables)
3. Expanded schema (additional functionality)
4. Engineering patterns schema
5. Index creation

This ensures proper database initialization and maintains all functionality from the original deployment scripts while adding new features.

#### WireGuard Integration
Convert VPN setup to container configuration:

```yaml
# File: docker-compose.yml (WireGuard section)
services:
  wireguard:
    image: linuxserver/wireguard
    cap_add:
      - NET_ADMIN
      - SYS_MODULE
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=UTC
      - SERVERURL=auto
      - SERVERPORT=51820
      - PEERS=1
      - PEERDNS=auto
      - INTERNAL_SUBNET=10.0.0.0/24
    volumes:
      - ./wireguard:/config
      - /lib/modules:/lib/modules:ro
    ports:
      - "51820:51820/udp"
    sysctls:
      - net.ipv4.conf.all.src_valid_mark=1
    restart: unless-stopped
```

#### Environment Configuration
Convert environment setup to `.env` file:

```env
# File: .env
# Database Configuration
POSTGRES_USER=aletheia_user
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=aletheia_db
POSTGRES_HOST_AUTH_METHOD=scram-sha-256  # Secure authentication method

# API Keys and External Services
OPENAI_API_KEY=your_api_key

# Hardware Configuration
GPU_DEVICE_IDS=0,1  # Maps to 5090 and 4090
NVIDIA_VISIBLE_DEVICES=all  # For container GPU access
NVIDIA_DRIVER_CAPABILITIES=compute,utility  # Required capabilities

# HearthMinds Network
HEARTHMINDS_SUBNET=10.10.0.0/24
ALETHEIA_PROXY_PORT=9000
LOGOS_PROXY_PORT=9001

# WireGuard
WIREGUARD_PORT=51820
```

### Complete Docker Compose Configuration

```yaml
# File: docker-compose.yml
version: '3.8'

services:
  aletheia-api:
    build:
      context: .
      dockerfile: Dockerfile.aletheia
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DB_CONNECTION=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@aletheia-db:5432/${POSTGRES_DB}
    volumes:
      - ./app:/app
    depends_on:
      - aletheia-db
      - aletheia-proxy
    networks:
      - aletheia-net
      - hearthminds-net

  aletheia-db:
    build:
      context: ./db
      dockerfile: Dockerfile.postgres
    environment:
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=${POSTGRES_DB}
    volumes:
      - aletheia-data:/var/lib/postgresql/data
    networks:
      - aletheia-net

  aletheia-proxy:
    build:
      context: .
      dockerfile: Dockerfile.proxy
    ports:
      - "${ALETHEIA_PROXY_PORT}:9000"
    networks:
      - hearthminds-net
    depends_on:
      - aletheia-api

  logos-api:
    build:
      context: .
      dockerfile: Dockerfile.logos
    environment:
      - CUDA_VISIBLE_DEVICES=${GPU_DEVICE_IDS}
      - DB_CONNECTION=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@logos-db:5432/${POSTGRES_DB}
    volumes:
      - ./logos:/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    depends_on:
      - logos-db
      - logos-proxy
    networks:
      - logos-net
      - hearthminds-net

  logos-db:
    build:
      context: ./db
      dockerfile: Dockerfile.postgres
    environment:
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=logos_db
    volumes:
      - logos-data:/var/lib/postgresql/data
    networks:
      - logos-net

  logos-proxy:
    build:
      context: .
      dockerfile: Dockerfile.proxy
    ports:
      - "${LOGOS_PROXY_PORT}:9000"
    networks:
      - hearthminds-net
    depends_on:
      - logos-api

  wireguard:
    image: linuxserver/wireguard
    cap_add:
      - NET_ADMIN
      - SYS_MODULE
    environment:
      - PUID=1000
      - PGID=1000
    volumes:
      - ./wireguard:/config
      - /lib/modules:/lib/modules:ro
    ports:
      - "${WIREGUARD_PORT}:51820/udp"
    sysctls:
      - net.ipv4.conf.all.src_valid_mark=1
    networks:
      - aletheia-net

volumes:
  aletheia-data:
  logos-data:

networks:
  aletheia-net:
    internal: true
  logos-net:
    internal: true
  hearthminds-net:
    ipam:
      config:
        - subnet: ${HEARTHMINDS_SUBNET}
```

### Deployment Process

1. **Initial Setup**:
   ```bash
   # Clone repository
   git clone <repo-url>
   cd <repo-dir>

   # Copy and edit environment configuration
   cp .env.example .env
   nano .env

   # Initialize WireGuard config
   mkdir -p wireguard
   ```

2. **Build and Deploy**:
   ```bash
   # Build all services
   docker-compose build

   # Start the stack
   docker-compose up -d

   # Monitor logs
   docker-compose logs -f
   ```

3. **Verify Deployment**:
   ```bash
   # Check service status
   docker-compose ps

   # Verify networks
   docker network ls | grep -E 'aletheia|logos|hearthminds'

   # Test database connections
   docker-compose exec aletheia-api python -c "from app.db import test_connection; test_connection()"
   ```

This containerized approach replaces the manual deployment scripts while maintaining all functionality and adding the HearthMinds network integration.