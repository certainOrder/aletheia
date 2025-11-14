# TODO_3: AI-Accessible Documentation System (Aletheia Self-Development)

**Purpose:** Enable Aletheia (and other AI agents) to access project-specific documentation patterns stored in PostgreSQL with semantic search, allowing her to help develop her own codebase with context-aware suggestions.

**Status:** Planning → Implementation → Testing → Production
**Priority:** High (enables self-directed development)
**Dependencies:** PostgreSQL with pgvector extension (✅ already installed)

---

## 📋 Executive Summary

The `eng_patterns` table in `hm_schema.sql` provides a **modular documentation system** where:
1. Engineering patterns/docs are stored as text with semantic embeddings
2. AI agents (Copilot, Aletheia, etc.) can query patterns via semantic search
3. Patterns are tagged and mapped to file contexts (glob patterns)
4. Retrieved patterns can be injected into AI context windows dynamically

**Key Insight:** Instead of static `.md` files, documentation lives in PostgreSQL with semantic search capabilities. AI agents query the database to retrieve relevant patterns based on:
- Current file being edited
- Semantic similarity to the task at hand
- Tags and strategy types

---

## 🎯 Goals

1. **AI Self-Service Documentation**: Aletheia can query her own documentation database
2. **Context-Aware Suggestions**: Retrieve patterns relevant to current file/task
3. **Semantic Search**: Find patterns by meaning, not just keywords
4. **Modular & Maintainable**: Add/update patterns without editing static files
5. **Multi-Agent Ready**: Shared knowledge base for all AI agents

---

## 📊 Current State (What Already Exists)

### Database Schema (✅ Implemented)
From `app/db/schema/hm_schema.sql`:

```sql
CREATE TABLE eng_patterns (
    id uuid PRIMARY KEY,
    content text NOT NULL,                    -- The pattern/documentation text
    tags text[],                              -- Array for tag filtering
    strategy_type text,                       -- 'design_pattern', 'coding_standard', etc.
    target_contexts text[],                   -- ['**/*_test.py', 'api/*.py']
    last_updated timestamp with time zone DEFAULT now(),
    author text,
    embedding vector(1536),                   -- OpenAI embeddings for semantic search
    metadata jsonb                            -- Flexible additional metadata
);

CREATE INDEX eng_patterns_embedding_idx ON eng_patterns 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX eng_patterns_tags_idx ON eng_patterns USING GIN (tags);
```

### Example Pattern (✅ Already Inserted)
A comprehensive "Boolean-Based Validation Testing Pattern" is already in the database, demonstrating:
- Markdown-formatted content with code examples
- Tags: `['testing', 'validation', 'best_practice', 'idempotency', 'python']`
- Target contexts: `['**/*_test.py', '**/*.py', 'app/db/**/*.py']`
- Strategy type: `'design_pattern'`

---

## 🏗️ Architecture: How AI Agents Access Documentation

### Flow Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│  AI Agent (Aletheia, Copilot, etc.)                            │
│  Current task: "Implement error handling for API endpoint"      │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      │ 1. Generate query embedding
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  OpenAI Embeddings API                                          │
│  Input: "error handling API endpoint"                           │
│  Output: vector(1536)                                           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      │ 2. Semantic search query
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  PostgreSQL (aletheia database)                                 │
│  SELECT content, tags, strategy_type                            │
│  FROM eng_patterns                                              │
│  WHERE target_contexts @> ARRAY['api/*.py']                     │
│  ORDER BY embedding <-> $query_embedding                        │
│  LIMIT 5                                                        │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      │ 3. Retrieved patterns (ranked by relevance)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Context Assembly                                               │
│  - Combine top N patterns into markdown                         │
│  - Add to AI context window                                     │
│  - AI generates code with pattern-aware suggestions             │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Pattern Storage** (✅ Done)
   - `eng_patterns` table with embeddings
   - Indexed for fast semantic search

2. **Embedding Generation** (⏳ TODO)
   - Service to generate embeddings for new patterns
   - Batch update script for existing patterns

3. **Query Interface** (⏳ TODO)
   - API endpoint: `POST /api/v1/context/search`
   - Python SDK: `ContextManager.query_patterns()`

4. **Context Assembly** (⏳ TODO)
   - Format retrieved patterns as markdown
   - Inject into AI context windows

5. **Pattern Management** (⏳ TODO)
   - CRUD operations for patterns
   - CLI tool for adding/updating patterns

---

## 🚀 Implementation Plan

### Phase 1: GitHub Copilot Context Generator (Week 1 - Quick Win!)

**Goal:** Generate semantic-based markdown files for GitHub Copilot consumption.

**Why First?** 
- ✅ No code changes to production app required
- ✅ Immediate value - Copilot can use patterns right away
- ✅ Validates pattern quality and workflow definitions
- ✅ Simple script, easy to iterate and refine

**Deliverable:** Working script that generates `.copilot/*.md` files from `eng_patterns` table.

#### Task 1.1: Semantic Context File Generator
**File:** `scripts/generate_copilot_contexts.py` (new file)

**Purpose:** Generate context-specific markdown files using **semantic similarity** instead of tags. This allows Copilot to access relevant patterns without real-time database queries.

**Key Features:**
- ✅ Uses embeddings for similarity-based retrieval (smarter than tags)
- ✅ Generates context files per workflow/file-pattern (e.g., `api_development.md`, `testing.md`)
- ✅ Automatically clusters related patterns
- ✅ Regenerates on pattern updates

```python
"""Generate context markdown files for GitHub Copilot using semantic clustering.

This script queries eng_patterns from the database and generates workflow-specific
markdown files based on semantic similarity rather than tags. Each generated file
contains patterns relevant to a specific development workflow (e.g., API development,
testing, database operations).

Usage:
    python scripts/generate_copilot_contexts.py
    python scripts/generate_copilot_contexts.py --output-dir .copilot
    python scripts/generate_copilot_contexts.py --workflows api testing db
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from sklearn.metrics.pairwise import cosine_similarity

from app.db.models import EngPattern
from app.db.setup.database import get_db
from app.services.openai_service import get_embedding
from app.logging_utils import logger


console = Console()


# Workflow definitions with representative queries
WORKFLOWS = {
    "api_development": {
        "query": "API endpoint implementation error handling validation routing",
        "description": "Patterns for building FastAPI endpoints and routes",
        "filename": "api_development.md",
    },
    "testing": {
        "query": "unit testing integration testing pytest fixtures mocking validation",
        "description": "Testing patterns and best practices",
        "filename": "testing.md",
    },
    "database": {
        "query": "database schema SQLAlchemy migrations queries models ORM",
        "description": "Database operations and schema management",
        "filename": "database.md",
    },
    "error_handling": {
        "query": "error handling exceptions logging debugging troubleshooting",
        "description": "Error handling and logging patterns",
        "filename": "error_handling.md",
    },
    "architecture": {
        "query": "system design architecture patterns services separation concerns",
        "description": "High-level architecture and design patterns",
        "filename": "architecture.md",
    },
    "deployment": {
        "query": "deployment docker containers CI/CD configuration environment",
        "description": "Deployment and infrastructure patterns",
        "filename": "deployment.md",
    },
}


class CopilotContextGenerator:
    """Generate semantic-based context files for GitHub Copilot."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db = next(get_db())

    def fetch_all_patterns(self) -> List[EngPattern]:
        """Fetch all patterns from database."""
        patterns = self.db.query(EngPattern).all()
        logger.info(f"Fetched {len(patterns)} patterns from database")
        return patterns

    def compute_workflow_embeddings(self) -> Dict[str, List[float]]:
        """Compute embeddings for each workflow query."""
        console.print("\n[cyan]Computing workflow embeddings...[/cyan]")
        workflow_embeddings = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating embeddings...", total=len(WORKFLOWS))
            
            for workflow_name, workflow_config in WORKFLOWS.items():
                query = workflow_config["query"]
                embedding = get_embedding(query)
                workflow_embeddings[workflow_name] = embedding
                progress.advance(task)
                logger.info(f"Generated embedding for workflow: {workflow_name}")
        
        return workflow_embeddings

    def assign_patterns_to_workflows(
        self,
        patterns: List[EngPattern],
        workflow_embeddings: Dict[str, List[float]],
        threshold: float = 0.7,
    ) -> Dict[str, List[Tuple[EngPattern, float]]]:
        """Assign patterns to workflows based on semantic similarity.
        
        Args:
            patterns: List of all patterns
            workflow_embeddings: Embeddings for each workflow
            threshold: Minimum similarity score (0-1) for inclusion
            
        Returns:
            Dict mapping workflow names to list of (pattern, similarity_score) tuples
        """
        console.print(f"\n[cyan]Assigning patterns to workflows (threshold: {threshold})...[/cyan]")
        
        workflow_patterns = defaultdict(list)
        
        for pattern in patterns:
            if pattern.embedding is None:
                logger.warning(f"Pattern {pattern.id} has no embedding, skipping")
                continue
            
            pattern_embedding = np.array(pattern.embedding).reshape(1, -1)
            
            # Calculate similarity to each workflow
            for workflow_name, workflow_embedding in workflow_embeddings.items():
                workflow_emb_array = np.array(workflow_embedding).reshape(1, -1)
                similarity = cosine_similarity(pattern_embedding, workflow_emb_array)[0][0]
                
                # Add pattern to workflow if similarity exceeds threshold
                if similarity >= threshold:
                    workflow_patterns[workflow_name].append((pattern, similarity))
        
        # Sort patterns by similarity within each workflow
        for workflow_name in workflow_patterns:
            workflow_patterns[workflow_name].sort(key=lambda x: x[1], reverse=True)
        
        # Log statistics
        for workflow_name, patterns_list in workflow_patterns.items():
            logger.info(f"{workflow_name}: {len(patterns_list)} patterns assigned")
        
        return workflow_patterns

    def generate_markdown_file(
        self,
        workflow_name: str,
        workflow_config: dict,
        patterns_with_scores: List[Tuple[EngPattern, float]],
    ) -> Path:
        """Generate a markdown file for a specific workflow.
        
        Args:
            workflow_name: Name of the workflow
            workflow_config: Workflow configuration dict
            patterns_with_scores: List of (pattern, similarity_score) tuples
            
        Returns:
            Path to generated file
        """
        output_file = self.output_dir / workflow_config["filename"]
        
        md_parts = []
        
        # Header
        md_parts.append(f"# {workflow_name.replace('_', ' ').title()} - Context for GitHub Copilot\n\n")
        md_parts.append(f"**Description:** {workflow_config['description']}\n\n")
        md_parts.append(f"**Auto-generated from:** `eng_patterns` database\n")
        md_parts.append(f"**Pattern count:** {len(patterns_with_scores)}\n")
        md_parts.append(f"**Relevance:** Patterns semantically matched to: _{workflow_config['query']}_\n\n")
        md_parts.append("---\n\n")
        
        # Table of contents
        md_parts.append("## Table of Contents\n\n")
        for idx, (pattern, score) in enumerate(patterns_with_scores, 1):
            pattern_title = pattern.strategy_type or f"Pattern {idx}"
            md_parts.append(f"{idx}. [{pattern_title}](#{self._slugify(pattern_title)}) "
                          f"(similarity: {score:.2f})\n")
        md_parts.append("\n---\n\n")
        
        # Patterns
        for idx, (pattern, score) in enumerate(patterns_with_scores, 1):
            pattern_title = pattern.strategy_type or f"Pattern {idx}"
            
            md_parts.append(f"## {idx}. {pattern_title}\n\n")
            md_parts.append(f"**Relevance Score:** {score:.3f} (cosine similarity)\n")
            md_parts.append(f"**Tags:** {', '.join(pattern.tags or [])}\n")
            md_parts.append(f"**Target Contexts:** {', '.join(pattern.target_contexts or [])}\n")
            md_parts.append(f"**Author:** {pattern.author or 'Unknown'}\n\n")
            md_parts.append(f"{pattern.content}\n\n")
            md_parts.append("---\n\n")
        
        # Footer
        md_parts.append("\n## How to Use This Context\n\n")
        md_parts.append("This file is automatically generated for GitHub Copilot. When working on ")
        md_parts.append(f"{workflow_name.replace('_', ' ')}, Copilot will use these patterns to:\n\n")
        md_parts.append("1. Suggest code that follows project conventions\n")
        md_parts.append("2. Provide inline documentation references\n")
        md_parts.append("3. Auto-complete boilerplate based on established patterns\n\n")
        md_parts.append("**To regenerate:** Run `python scripts/generate_copilot_contexts.py`\n")
        
        # Write to file
        output_file.write_text("".join(md_parts))
        logger.info(f"Generated: {output_file} ({len(patterns_with_scores)} patterns)")
        
        return output_file

    def generate_index(self, generated_files: List[Path]):
        """Generate an index file listing all context files."""
        index_file = self.output_dir / "README.md"
        
        md_parts = []
        md_parts.append("# GitHub Copilot Context Files\n\n")
        md_parts.append("This directory contains auto-generated context files for GitHub Copilot. ")
        md_parts.append("Each file provides workflow-specific patterns and documentation.\n\n")
        md_parts.append("## Available Contexts\n\n")
        
        for file_path in sorted(generated_files):
            workflow_name = file_path.stem.replace("_", " ").title()
            md_parts.append(f"- [{workflow_name}]({file_path.name})\n")
        
        md_parts.append("\n## How It Works\n\n")
        md_parts.append("1. Patterns stored in `eng_patterns` database table\n")
        md_parts.append("2. Script generates embeddings for workflow queries\n")
        md_parts.append("3. Patterns assigned to workflows via semantic similarity\n")
        md_parts.append("4. Context files regenerated when patterns change\n\n")
        md_parts.append("## Regeneration\n\n")
        md_parts.append("```bash\n")
        md_parts.append("python scripts/generate_copilot_contexts.py\n")
        md_parts.append("```\n\n")
        md_parts.append("**Last generated:** Check file timestamps\n")
        
        index_file.write_text("".join(md_parts))
        logger.info(f"Generated index: {index_file}")

    def _slugify(self, text: str) -> str:
        """Convert text to markdown anchor slug."""
        return text.lower().replace(" ", "-").replace("_", "-")

    def generate_all(self, workflows: Dict[str, dict] = None, threshold: float = 0.7):
        """Generate all context files.
        
        Args:
            workflows: Dict of workflow definitions (uses WORKFLOWS if None)
            threshold: Minimum similarity threshold for pattern inclusion
        """
        workflows = workflows or WORKFLOWS
        
        console.print("[bold green]Generating GitHub Copilot Context Files[/bold green]\n")
        
        # Fetch patterns
        patterns = self.fetch_all_patterns()
        
        if not patterns:
            console.print("[yellow]⚠ No patterns found in database[/yellow]")
            return
        
        # Compute workflow embeddings
        workflow_embeddings = self.compute_workflow_embeddings()
        
        # Assign patterns to workflows
        workflow_patterns = self.assign_patterns_to_workflows(
            patterns, workflow_embeddings, threshold
        )
        
        # Generate markdown files
        console.print("\n[cyan]Generating markdown files...[/cyan]\n")
        generated_files = []
        
        for workflow_name, workflow_config in workflows.items():
            patterns_with_scores = workflow_patterns.get(workflow_name, [])
            
            if not patterns_with_scores:
                console.print(f"[yellow]⚠ No patterns for {workflow_name} (threshold too high?)[/yellow]")
                continue
            
            output_file = self.generate_markdown_file(
                workflow_name, workflow_config, patterns_with_scores
            )
            generated_files.append(output_file)
            console.print(f"[green]✓[/green] {output_file.name}: {len(patterns_with_scores)} patterns")
        
        # Generate index
        self.generate_index(generated_files)
        
        console.print(f"\n[bold green]✓ Generated {len(generated_files)} context files in {self.output_dir}[/bold green]")


def main():
    parser = argparse.ArgumentParser(
        description="Generate semantic-based context files for GitHub Copilot"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".copilot"),
        help="Output directory for context files (default: .copilot)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum similarity threshold (0-1, default: 0.7)",
    )
    parser.add_argument(
        "--workflows",
        nargs="+",
        help="Generate only specific workflows (e.g., api testing)",
    )
    
    args = parser.parse_args()
    
    # Filter workflows if specified
    workflows = WORKFLOWS
    if args.workflows:
        workflows = {k: v for k, v in WORKFLOWS.items() if k in args.workflows}
    
    # Generate
    generator = CopilotContextGenerator(args.output_dir)
    generator.generate_all(workflows=workflows, threshold=args.threshold)


if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# Generate all context files
python scripts/generate_copilot_contexts.py

# Output to custom directory
python scripts/generate_copilot_contexts.py --output-dir docs/copilot_contexts

# Generate only specific workflows
python scripts/generate_copilot_contexts.py --workflows api testing

# Adjust similarity threshold (lower = more patterns included)
python scripts/generate_copilot_contexts.py --threshold 0.6
```

**Output Structure:**
```
.copilot/
├── README.md                 # Index of all context files
├── api_development.md        # API patterns (sorted by relevance)
├── testing.md                # Testing patterns
├── database.md               # Database patterns
├── error_handling.md         # Error handling patterns
├── architecture.md           # Architecture patterns
└── deployment.md             # Deployment patterns
```

**Dependencies:**
Add to `requirements-dev.txt`:
```txt
scikit-learn>=1.3.0  # For cosine_similarity
numpy>=1.24.0
```

#### Task 1.2: Add EngPattern Model
**File:** `app/db/models.py`

Add this model to query the `eng_patterns` table:

```python
class EngPattern(Base):
    """Engineering pattern/documentation with semantic embeddings.
    
    Stores modular documentation that AI agents can query via semantic search.
    Patterns are tagged, mapped to file contexts, and indexed for fast retrieval.
    """
    __tablename__ = "eng_patterns"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tags: Mapped[Optional[list[str]]] = mapped_column(ARRAY(Text), nullable=True)
    strategy_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    target_contexts: Mapped[Optional[list[str]]] = mapped_column(
        ARRAY(Text), nullable=True
    )
    last_updated: Mapped[Optional[str]] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    author: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )
    metadata_json: Mapped[Optional[dict]] = mapped_column(
        "metadata", JSONB, nullable=True
    )

    def __repr__(self) -> str:
        return f"<EngPattern id={self.id} type={self.strategy_type}>"
```

**Validation:**
```python
# Test in Python REPL
from app.db.models import EngPattern
from app.db.setup.database import get_db

db = next(get_db())
pattern = db.query(EngPattern).first()
print(pattern.content[:100])  # Should print first 100 chars of pattern
```

#### Task 1.3: Integration with Makefile & Git
**File:** `Makefile`

Add target for easy regeneration:
```makefile
generate-copilot-contexts:
	@echo "Generating Copilot context files from eng_patterns database..."
	python scripts/generate_copilot_contexts.py --output-dir .copilot
	@echo "✓ Context files generated in .copilot/"

.PHONY: generate-copilot-contexts
```

**File:** `.gitignore`

Optionally track generated files (or ignore them):
```gitignore
# Option 1: Track generated files (recommended for visibility)
# .copilot/  # Leave commented to track files

# Option 2: Ignore and regenerate on each machine
# .copilot/  # Uncomment to ignore
```

**Usage:**
```bash
# Generate contexts
make generate-copilot-contexts

# Now Copilot can read .copilot/*.md files
```

#### Task 1.4: Validate Output

**Test the script:**
```bash
# 1. Ensure you have the required dependencies
pip install scikit-learn numpy rich

# 2. Run the script
python scripts/generate_copilot_contexts.py

# 3. Check output
ls -lh .copilot/
cat .copilot/README.md
head -50 .copilot/testing.md
```

**Expected output:**
```
Generating GitHub Copilot Context Files

Computing workflow embeddings...
✓ api_development.md: 3 patterns
✓ testing.md: 5 patterns
✓ database.md: 2 patterns
...

✓ Generated 6 context files in .copilot
```

---

### Phase 2: SQLAlchemy Model & Context Service (Week 1-2)

**Goal:** Create Python interface to query `eng_patterns` table (for Aletheia's direct access).

**Why Second?** Now that Copilot integration is working, we can build the programmatic API that Aletheia will use.

#### Task 2.1: Context Service
**File:** `app/services/context_service.py` (new file)

```python
"""Context service for retrieving AI-relevant documentation patterns."""

import uuid
from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import EngPattern
from app.services.openai_service import get_embedding
from app.logging_utils import logger


class ContextService:
    """Service for querying engineering patterns with semantic search."""

    def __init__(self, db: Session):
        self.db = db

    def search_patterns(
        self,
        query: str,
        file_pattern: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[EngPattern]:
        """Search patterns using semantic similarity.
        
        Args:
            query: Natural language query (e.g., "error handling patterns")
            file_pattern: Filter by target context (e.g., "api/*.py")
            tags: Filter by tags (e.g., ["error_handling", "api"])
            limit: Maximum number of results
            
        Returns:
            List of EngPattern objects ranked by semantic similarity
        """
        logger.info(f"Searching patterns: query='{query}', file_pattern={file_pattern}")
        
        # Generate embedding for query
        query_embedding = get_embedding(query)
        
        # Build query
        stmt = select(EngPattern)
        
        # Filter by file pattern if provided
        if file_pattern:
            stmt = stmt.where(EngPattern.target_contexts.contains([file_pattern]))
        
        # Filter by tags if provided
        if tags:
            stmt = stmt.where(EngPattern.tags.overlap(tags))
        
        # Order by semantic similarity (cosine distance)
        stmt = stmt.order_by(
            EngPattern.embedding.cosine_distance(query_embedding)
        ).limit(limit)
        
        results = self.db.execute(stmt).scalars().all()
        logger.info(f"Found {len(results)} matching patterns")
        return results

    def get_pattern_by_id(self, pattern_id: uuid.UUID) -> Optional[EngPattern]:
        """Retrieve a specific pattern by ID."""
        return self.db.query(EngPattern).filter(EngPattern.id == pattern_id).first()

    def list_target_contexts(self) -> List[str]:
        """List all unique target contexts (file patterns)."""
        stmt = select(func.unnest(EngPattern.target_contexts)).distinct()
        return [row[0] for row in self.db.execute(stmt).all()]

    def list_tags(self) -> List[str]:
        """List all unique tags."""
        stmt = select(func.unnest(EngPattern.tags)).distinct()
        return [row[0] for row in self.db.execute(stmt).all()]

    def format_as_markdown(self, patterns: List[EngPattern]) -> str:
        """Format retrieved patterns as markdown for AI context.
        
        Args:
            patterns: List of EngPattern objects
            
        Returns:
            Formatted markdown string ready for AI consumption
        """
        if not patterns:
            return "# No Relevant Patterns Found\n\n"
        
        md_parts = ["# Engineering Patterns (Context)\n\n"]
        
        for idx, pattern in enumerate(patterns, 1):
            md_parts.append(f"## Pattern {idx}: {pattern.strategy_type or 'General'}\n")
            md_parts.append(f"**Tags:** {', '.join(pattern.tags or [])}\n\n")
            md_parts.append(f"{pattern.content}\n\n")
            md_parts.append("---\n\n")
        
        return "".join(md_parts)
```

**Validation:**
```python
# Test semantic search
from app.services.context_service import ContextService
from app.db.setup.database import get_db

db = next(get_db())
cs = ContextService(db)

# Search for validation patterns
patterns = cs.search_patterns("validation testing", file_pattern="**/*_test.py")
print(f"Found {len(patterns)} patterns")

# Format as markdown
md = cs.format_as_markdown(patterns)
print(md[:500])
```

---

### Phase 3: API Endpoints for AI Agents (Week 2)

**Goal:** Expose REST API for Aletheia and other AI agents to query patterns.

**Why Third?** With the service layer complete, we expose it via REST API for programmatic access.

#### Task 2.1: Create API Schema
**File:** `app/schemas/context.py` (new file)

```python
"""Pydantic schemas for context/pattern API."""

from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class PatternSearchRequest(BaseModel):
    """Request to search for patterns."""
    query: str = Field(..., description="Natural language query")
    file_pattern: Optional[str] = Field(None, description="Filter by target context")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: int = Field(5, ge=1, le=20, description="Max results")


class PatternSearchResponse(BaseModel):
    """Response containing matched patterns."""
    query: str
    patterns: List["PatternDetail"]
    total_results: int


class PatternDetail(BaseModel):
    """Details of a single pattern."""
    id: UUID
    content: str
    tags: List[str]
    strategy_type: Optional[str]
    target_contexts: List[str]
    similarity_score: Optional[float] = Field(
        None, description="Cosine similarity score (0-1)"
    )


class ContextMarkdownResponse(BaseModel):
    """Formatted markdown context for AI consumption."""
    query: str
    markdown: str
    pattern_count: int
```

#### Task 2.2: Add API Endpoint
**File:** `app/api/routes.py`

Add to existing routes:

```python
from app.services.context_service import ContextService
from app.schemas.context import (
    PatternSearchRequest,
    PatternSearchResponse,
    ContextMarkdownResponse,
)

@router.post("/context/search", response_model=PatternSearchResponse)
def search_patterns(
    request: PatternSearchRequest,
    db: Session = Depends(get_db),
) -> PatternSearchResponse:
    """Search engineering patterns using semantic similarity.
    
    AI agents can use this endpoint to retrieve relevant documentation
    patterns based on natural language queries and file contexts.
    """
    cs = ContextService(db)
    patterns = cs.search_patterns(
        query=request.query,
        file_pattern=request.file_pattern,
        tags=request.tags,
        limit=request.limit,
    )
    
    return PatternSearchResponse(
        query=request.query,
        patterns=[
            PatternDetail(
                id=p.id,
                content=p.content,
                tags=p.tags or [],
                strategy_type=p.strategy_type,
                target_contexts=p.target_contexts or [],
            )
            for p in patterns
        ],
        total_results=len(patterns),
    )


@router.post("/context/markdown", response_model=ContextMarkdownResponse)
def get_context_markdown(
    request: PatternSearchRequest,
    db: Session = Depends(get_db),
) -> ContextMarkdownResponse:
    """Get patterns formatted as markdown for AI context injection.
    
    Returns assembled markdown ready to be added to an AI's context window.
    """
    cs = ContextService(db)
    patterns = cs.search_patterns(
        query=request.query,
        file_pattern=request.file_pattern,
        tags=request.tags,
        limit=request.limit,
    )
    
    markdown = cs.format_as_markdown(patterns)
    
    return ContextMarkdownResponse(
        query=request.query,
        markdown=markdown,
        pattern_count=len(patterns),
    )
```

**Validation:**
```bash
# Test the API endpoint
curl -X POST http://localhost:8000/api/v1/context/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "validation testing patterns",
    "file_pattern": "**/*_test.py",
    "limit": 3
  }'

# Get markdown format
curl -X POST http://localhost:8000/api/v1/context/markdown \
  -H "Content-Type: application/json" \
  -d '{
    "query": "error handling",
    "tags": ["api", "error_handling"],
    "limit": 5
  }'
```

---

### Phase 4: Pattern Management Tools (Week 2-3)

**Goal:** Tools to add, update, and manage patterns.

**Why Fourth?** Now that the system is working end-to-end, build tooling to maintain pattern quality.

#### Task 3.1: Pattern Management CLI
**File:** `app/services/pattern_manager.py` (new file)

```python
"""CLI tool for managing engineering patterns."""

import sys
import uuid
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table
from sqlalchemy.orm import Session

from app.db.models import EngPattern
from app.db.setup.database import get_db
from app.services.openai_service import get_embedding
from app.logging_utils import logger


console = Console()


class PatternManager:
    """Manager for CRUD operations on engineering patterns."""

    def __init__(self, db: Session):
        self.db = db

    def add_pattern(
        self,
        content: str,
        tags: List[str],
        strategy_type: str,
        target_contexts: List[str],
        author: str = "system",
        metadata: Optional[dict] = None,
    ) -> EngPattern:
        """Add a new pattern to the database.
        
        Args:
            content: Markdown-formatted pattern content
            tags: List of tags
            strategy_type: Type of pattern (e.g., 'design_pattern')
            target_contexts: File glob patterns (e.g., ['api/*.py'])
            author: Pattern author
            metadata: Additional metadata
            
        Returns:
            Created EngPattern object
        """
        logger.info(f"Adding pattern: type={strategy_type}, tags={tags}")
        
        # Generate embedding
        embedding = get_embedding(content)
        
        pattern = EngPattern(
            id=uuid.uuid4(),
            content=content,
            tags=tags,
            strategy_type=strategy_type,
            target_contexts=target_contexts,
            author=author,
            embedding=embedding,
            metadata_json=metadata,
        )
        
        self.db.add(pattern)
        self.db.commit()
        self.db.refresh(pattern)
        
        logger.info(f"Pattern added: id={pattern.id}")
        return pattern

    def update_pattern(
        self,
        pattern_id: uuid.UUID,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        strategy_type: Optional[str] = None,
        target_contexts: Optional[List[str]] = None,
    ) -> Optional[EngPattern]:
        """Update an existing pattern."""
        pattern = self.db.query(EngPattern).filter(EngPattern.id == pattern_id).first()
        if not pattern:
            logger.warning(f"Pattern not found: id={pattern_id}")
            return None
        
        if content is not None:
            pattern.content = content
            pattern.embedding = get_embedding(content)
        
        if tags is not None:
            pattern.tags = tags
        
        if strategy_type is not None:
            pattern.strategy_type = strategy_type
        
        if target_contexts is not None:
            pattern.target_contexts = target_contexts
        
        self.db.commit()
        self.db.refresh(pattern)
        logger.info(f"Pattern updated: id={pattern_id}")
        return pattern

    def delete_pattern(self, pattern_id: uuid.UUID) -> bool:
        """Delete a pattern."""
        pattern = self.db.query(EngPattern).filter(EngPattern.id == pattern_id).first()
        if not pattern:
            return False
        
        self.db.delete(pattern)
        self.db.commit()
        logger.info(f"Pattern deleted: id={pattern_id}")
        return True

    def list_patterns(self) -> List[EngPattern]:
        """List all patterns."""
        return self.db.query(EngPattern).all()

    def import_from_markdown(self, file_path: Path) -> EngPattern:
        """Import a pattern from a markdown file.
        
        Expects YAML frontmatter with metadata:
        ---
        tags: [testing, validation]
        strategy_type: design_pattern
        target_contexts: ["**/*_test.py"]
        author: copilot
        ---
        
        # Pattern Title
        Pattern content here...
        """
        import yaml
        
        content = file_path.read_text()
        
        # Parse frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            frontmatter = yaml.safe_load(parts[1])
            pattern_content = parts[2].strip()
        else:
            raise ValueError("Markdown file must have YAML frontmatter")
        
        return self.add_pattern(
            content=pattern_content,
            tags=frontmatter.get("tags", []),
            strategy_type=frontmatter.get("strategy_type", "general"),
            target_contexts=frontmatter.get("target_contexts", []),
            author=frontmatter.get("author", "system"),
            metadata=frontmatter.get("metadata"),
        )


@click.group()
def cli():
    """Engineering pattern management CLI."""
    pass


@cli.command()
def list_patterns():
    """List all patterns."""
    db = next(get_db())
    pm = PatternManager(db)
    patterns = pm.list_patterns()
    
    table = Table(title="Engineering Patterns")
    table.add_column("ID", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Tags", style="yellow")
    table.add_column("Contexts", style="magenta")
    
    for p in patterns:
        table.add_row(
            str(p.id)[:8],
            p.strategy_type or "N/A",
            ", ".join(p.tags or []),
            ", ".join(p.target_contexts or [])[:50],
        )
    
    console.print(table)


@cli.command()
@click.argument("markdown_file", type=click.Path(exists=True))
def import_pattern(markdown_file):
    """Import a pattern from a markdown file."""
    db = next(get_db())
    pm = PatternManager(db)
    
    try:
        pattern = pm.import_from_markdown(Path(markdown_file))
        console.print(f"[green]✓[/green] Pattern imported: {pattern.id}")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to import: {e}")
        sys.exit(1)


@cli.command()
@click.argument("pattern_id")
def delete_pattern(pattern_id):
    """Delete a pattern by ID."""
    db = next(get_db())
    pm = PatternManager(db)
    
    if pm.delete_pattern(uuid.UUID(pattern_id)):
        console.print(f"[green]✓[/green] Pattern deleted: {pattern_id}")
    else:
        console.print(f"[red]✗[/red] Pattern not found: {pattern_id}")


if __name__ == "__main__":
    cli()
```

**Usage:**
```bash
# List all patterns
python -m app.services.pattern_manager list-patterns

# Import a new pattern from markdown
python -m app.services.pattern_manager import-pattern docs/patterns/api_error_handling.md

# Delete a pattern
python -m app.services.pattern_manager delete-pattern <uuid>
```

#### Task 3.2: Batch Embedding Generator
**File:** `scripts/generate_pattern_embeddings.py` (new file)

```python
"""Batch generate embeddings for patterns without embeddings."""

from app.db.models import EngPattern
from app.db.setup.database import get_db
from app.services.openai_service import get_embedding
from app.logging_utils import logger


def generate_missing_embeddings():
    """Generate embeddings for patterns that don't have them."""
    db = next(get_db())
    
    # Find patterns without embeddings
    patterns = db.query(EngPattern).filter(EngPattern.embedding == None).all()
    
    logger.info(f"Found {len(patterns)} patterns without embeddings")
    
    for pattern in patterns:
        logger.info(f"Generating embedding for pattern {pattern.id}")
        try:
            pattern.embedding = get_embedding(pattern.content)
            db.commit()
            logger.info(f"✓ Embedding generated for {pattern.id}")
        except Exception as e:
            logger.error(f"✗ Failed to generate embedding for {pattern.id}: {e}")
            db.rollback()
    
    logger.info("Embedding generation complete")


if __name__ == "__main__":
    generate_missing_embeddings()
```

**Usage:**
```bash
python scripts/generate_pattern_embeddings.py
```

---

### Phase 5: Aletheia Context Injection (Week 3-4)

**Goal:** Enable Aletheia to automatically query and use patterns.

**Why Fifth?** With all infrastructure in place, integrate with Aletheia's workflow.

#### Task 5.1: Aletheia Context Injection

**Concept:** When Aletheia is working on code, she can query relevant patterns and inject them into her context window.

**Example Workflow:**

This script queries eng_patterns from the database and generates workflow-specific
markdown files based on semantic similarity rather than tags. Each generated file
contains patterns relevant to a specific development workflow (e.g., API development,
testing, database operations).

Usage:
    python scripts/generate_copilot_contexts.py
    python scripts/generate_copilot_contexts.py --output-dir .copilot
    python scripts/generate_copilot_contexts.py --workflows api testing db
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from sklearn.metrics.pairwise import cosine_similarity

from app.db.models import EngPattern
from app.db.setup.database import get_db
from app.services.openai_service import get_embedding
from app.logging_utils import logger


console = Console()


# Workflow definitions with representative queries
WORKFLOWS = {
    "api_development": {
        "query": "API endpoint implementation error handling validation routing",
        "description": "Patterns for building FastAPI endpoints and routes",
        "filename": "api_development.md",
    },
    "testing": {
        "query": "unit testing integration testing pytest fixtures mocking validation",
        "description": "Testing patterns and best practices",
        "filename": "testing.md",
    },
    "database": {
        "query": "database schema SQLAlchemy migrations queries models ORM",
        "description": "Database operations and schema management",
        "filename": "database.md",
    },
    "error_handling": {
        "query": "error handling exceptions logging debugging troubleshooting",
        "description": "Error handling and logging patterns",
        "filename": "error_handling.md",
    },
    "architecture": {
        "query": "system design architecture patterns services separation concerns",
        "description": "High-level architecture and design patterns",
        "filename": "architecture.md",
    },
    "deployment": {
        "query": "deployment docker containers CI/CD configuration environment",
        "description": "Deployment and infrastructure patterns",
        "filename": "deployment.md",
    },
}


class CopilotContextGenerator:
    """Generate semantic-based context files for GitHub Copilot."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db = next(get_db())

    def fetch_all_patterns(self) -> List[EngPattern]:
        """Fetch all patterns from database."""
        patterns = self.db.query(EngPattern).all()
        logger.info(f"Fetched {len(patterns)} patterns from database")
        return patterns

    def compute_workflow_embeddings(self) -> Dict[str, List[float]]:
        """Compute embeddings for each workflow query."""
        console.print("\n[cyan]Computing workflow embeddings...[/cyan]")
        workflow_embeddings = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating embeddings...", total=len(WORKFLOWS))
            
            for workflow_name, workflow_config in WORKFLOWS.items():
                query = workflow_config["query"]
                embedding = get_embedding(query)
                workflow_embeddings[workflow_name] = embedding
                progress.advance(task)
                logger.info(f"Generated embedding for workflow: {workflow_name}")
        
        return workflow_embeddings

    def assign_patterns_to_workflows(
        self,
        patterns: List[EngPattern],
        workflow_embeddings: Dict[str, List[float]],
        threshold: float = 0.7,
    ) -> Dict[str, List[Tuple[EngPattern, float]]]:
        """Assign patterns to workflows based on semantic similarity.
        
        Args:
            patterns: List of all patterns
            workflow_embeddings: Embeddings for each workflow
            threshold: Minimum similarity score (0-1) for inclusion
            
        Returns:
            Dict mapping workflow names to list of (pattern, similarity_score) tuples
        """
        console.print(f"\n[cyan]Assigning patterns to workflows (threshold: {threshold})...[/cyan]")
        
        workflow_patterns = defaultdict(list)
        
        for pattern in patterns:
            if pattern.embedding is None:
                logger.warning(f"Pattern {pattern.id} has no embedding, skipping")
                continue
            
            pattern_embedding = np.array(pattern.embedding).reshape(1, -1)
            
            # Calculate similarity to each workflow
            for workflow_name, workflow_embedding in workflow_embeddings.items():
                workflow_emb_array = np.array(workflow_embedding).reshape(1, -1)
                similarity = cosine_similarity(pattern_embedding, workflow_emb_array)[0][0]
                
                # Add pattern to workflow if similarity exceeds threshold
                if similarity >= threshold:
                    workflow_patterns[workflow_name].append((pattern, similarity))
        
        # Sort patterns by similarity within each workflow
        for workflow_name in workflow_patterns:
            workflow_patterns[workflow_name].sort(key=lambda x: x[1], reverse=True)
        
        # Log statistics
        for workflow_name, patterns_list in workflow_patterns.items():
            logger.info(f"{workflow_name}: {len(patterns_list)} patterns assigned")
        
        return workflow_patterns

    def generate_markdown_file(
        self,
        workflow_name: str,
        workflow_config: dict,
        patterns_with_scores: List[Tuple[EngPattern, float]],
    ) -> Path:
        """Generate a markdown file for a specific workflow.
        
        Args:
            workflow_name: Name of the workflow
            workflow_config: Workflow configuration dict
            patterns_with_scores: List of (pattern, similarity_score) tuples
            
        Returns:
            Path to generated file
        """
        output_file = self.output_dir / workflow_config["filename"]
        
        md_parts = []
        
        # Header
        md_parts.append(f"# {workflow_name.replace('_', ' ').title()} - Context for GitHub Copilot\n\n")
        md_parts.append(f"**Description:** {workflow_config['description']}\n\n")
        md_parts.append(f"**Auto-generated from:** `eng_patterns` database\n")
        md_parts.append(f"**Pattern count:** {len(patterns_with_scores)}\n")
        md_parts.append(f"**Relevance:** Patterns semantically matched to: _{workflow_config['query']}_\n\n")
        md_parts.append("---\n\n")
        
        # Table of contents
        md_parts.append("## Table of Contents\n\n")
        for idx, (pattern, score) in enumerate(patterns_with_scores, 1):
            pattern_title = pattern.strategy_type or f"Pattern {idx}"
            md_parts.append(f"{idx}. [{pattern_title}](#{self._slugify(pattern_title)}) "
                          f"(similarity: {score:.2f})\n")
        md_parts.append("\n---\n\n")
        
        # Patterns
        for idx, (pattern, score) in enumerate(patterns_with_scores, 1):
            pattern_title = pattern.strategy_type or f"Pattern {idx}"
            
            md_parts.append(f"## {idx}. {pattern_title}\n\n")
            md_parts.append(f"**Relevance Score:** {score:.3f} (cosine similarity)\n")
            md_parts.append(f"**Tags:** {', '.join(pattern.tags or [])}\n")
            md_parts.append(f"**Target Contexts:** {', '.join(pattern.target_contexts or [])}\n")
            md_parts.append(f"**Author:** {pattern.author or 'Unknown'}\n\n")
            md_parts.append(f"{pattern.content}\n\n")
            md_parts.append("---\n\n")
        
        # Footer
        md_parts.append("\n## How to Use This Context\n\n")
        md_parts.append("This file is automatically generated for GitHub Copilot. When working on ")
        md_parts.append(f"{workflow_name.replace('_', ' ')}, Copilot will use these patterns to:\n\n")
        md_parts.append("1. Suggest code that follows project conventions\n")
        md_parts.append("2. Provide inline documentation references\n")
        md_parts.append("3. Auto-complete boilerplate based on established patterns\n\n")
        md_parts.append("**To regenerate:** Run `python scripts/generate_copilot_contexts.py`\n")
        
        # Write to file
        output_file.write_text("".join(md_parts))
        logger.info(f"Generated: {output_file} ({len(patterns_with_scores)} patterns)")
        
        return output_file

    def generate_index(self, generated_files: List[Path]):
        """Generate an index file listing all context files."""
        index_file = self.output_dir / "README.md"
        
        md_parts = []
        md_parts.append("# GitHub Copilot Context Files\n\n")
        md_parts.append("This directory contains auto-generated context files for GitHub Copilot. ")
        md_parts.append("Each file provides workflow-specific patterns and documentation.\n\n")
        md_parts.append("## Available Contexts\n\n")
        
        for file_path in sorted(generated_files):
            workflow_name = file_path.stem.replace("_", " ").title()
            md_parts.append(f"- [{workflow_name}]({file_path.name})\n")
        
        md_parts.append("\n## How It Works\n\n")
        md_parts.append("1. Patterns stored in `eng_patterns` database table\n")
        md_parts.append("2. Script generates embeddings for workflow queries\n")
        md_parts.append("3. Patterns assigned to workflows via semantic similarity\n")
        md_parts.append("4. Context files regenerated when patterns change\n\n")
        md_parts.append("## Regeneration\n\n")
        md_parts.append("```bash\n")
        md_parts.append("python scripts/generate_copilot_contexts.py\n")
        md_parts.append("```\n\n")
        md_parts.append("**Last generated:** Check file timestamps\n")
        
        index_file.write_text("".join(md_parts))
        logger.info(f"Generated index: {index_file}")

    def _slugify(self, text: str) -> str:
        """Convert text to markdown anchor slug."""
        return text.lower().replace(" ", "-").replace("_", "-")

    def generate_all(self, workflows: Dict[str, dict] = None, threshold: float = 0.7):
        """Generate all context files.
        
        Args:
            workflows: Dict of workflow definitions (uses WORKFLOWS if None)
            threshold: Minimum similarity threshold for pattern inclusion
        """
        workflows = workflows or WORKFLOWS
        
        console.print("[bold green]Generating GitHub Copilot Context Files[/bold green]\n")
        
        # Fetch patterns
        patterns = self.fetch_all_patterns()
        
        if not patterns:
            console.print("[yellow]⚠ No patterns found in database[/yellow]")
            return
        
        # Compute workflow embeddings
        workflow_embeddings = self.compute_workflow_embeddings()
        
        # Assign patterns to workflows
        workflow_patterns = self.assign_patterns_to_workflows(
            patterns, workflow_embeddings, threshold
        )
        
        # Generate markdown files
        console.print("\n[cyan]Generating markdown files...[/cyan]\n")
        generated_files = []
        
        for workflow_name, workflow_config in workflows.items():
            patterns_with_scores = workflow_patterns.get(workflow_name, [])
            
            if not patterns_with_scores:
                console.print(f"[yellow]⚠ No patterns for {workflow_name} (threshold too high?)[/yellow]")
                continue
            
            output_file = self.generate_markdown_file(
                workflow_name, workflow_config, patterns_with_scores
            )
            generated_files.append(output_file)
            console.print(f"[green]✓[/green] {output_file.name}: {len(patterns_with_scores)} patterns")
        
        # Generate index
        self.generate_index(generated_files)
        
        console.print(f"\n[bold green]✓ Generated {len(generated_files)} context files in {self.output_dir}[/bold green]")


def main():
    parser = argparse.ArgumentParser(
        description="Generate semantic-based context files for GitHub Copilot"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".copilot"),
        help="Output directory for context files (default: .copilot)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum similarity threshold (0-1, default: 0.7)",
    )
    parser.add_argument(
        "--workflows",
        nargs="+",
        help="Generate only specific workflows (e.g., api testing)",
    )
    
    args = parser.parse_args()
    
    # Filter workflows if specified
    workflows = WORKFLOWS
    if args.workflows:
        workflows = {k: v for k, v in WORKFLOWS.items() if k in args.workflows}
    
    # Generate
    generator = CopilotContextGenerator(args.output_dir)
    generator.generate_all(workflows=workflows, threshold=args.threshold)


if __name__ == "__main__":
    main()
```

**Key Features:**

1. **Semantic Clustering:** Uses embeddings to match patterns to workflows
2. **Relevance Scores:** Each pattern includes similarity score (helps Copilot prioritize)
3. **Multiple Workflows:** Generates separate files for API, testing, DB, etc.
4. **Auto-indexing:** Creates README with links to all context files
5. **Configurable:** Adjust similarity threshold, workflows, output directory

**Workflow Definitions:**

Each workflow has:
- **Query:** Semantic description of the workflow (embedded for similarity)
- **Description:** Human-readable explanation
- **Filename:** Output markdown file name

Add new workflows by extending the `WORKFLOWS` dict:
```python
WORKFLOWS["security"] = {
    "query": "authentication authorization security validation JWT tokens",
    "description": "Security and auth patterns",
    "filename": "security.md",
}
```

**Usage:**
```bash
# Generate all context files
python scripts/generate_copilot_contexts.py

# Output to custom directory
python scripts/generate_copilot_contexts.py --output-dir docs/copilot_contexts

# Generate only specific workflows
python scripts/generate_copilot_contexts.py --workflows api testing

# Adjust similarity threshold (lower = more patterns included)
python scripts/generate_copilot_contexts.py --threshold 0.6
```

**Output Structure:**
```
.copilot/
├── README.md                 # Index of all context files
├── api_development.md        # API patterns (sorted by relevance)
├── testing.md                # Testing patterns
├── database.md               # Database patterns
├── error_handling.md         # Error handling patterns
├── architecture.md           # Architecture patterns
└── deployment.md             # Deployment patterns
```

**Example Generated File (`api_development.md`):**
```markdown
# Api Development - Context for GitHub Copilot

**Description:** Patterns for building FastAPI endpoints and routes
**Auto-generated from:** `eng_patterns` database
**Pattern count:** 8
**Relevance:** Patterns semantically matched to: _API endpoint implementation error handling validation routing_

---

## Table of Contents

1. [API Error Handling](api-error-handling) (similarity: 0.89)
2. [Request Validation Pattern](request-validation-pattern) (similarity: 0.84)
3. [Endpoint Testing](endpoint-testing) (similarity: 0.78)
...

## 1. API Error Handling

**Relevance Score:** 0.891 (cosine similarity)
**Tags:** error_handling, api, fastapi
**Target Contexts:** api/*.py, routes/*.py
**Author:** copilot

[Pattern content here...]
```

**Integration with Makefile:**
Add to `Makefile`:
```makefile
generate-copilot-contexts:
	@echo "Generating Copilot context files..."
	python scripts/generate_copilot_contexts.py --output-dir .copilot
	@echo "✓ Context files generated in .copilot/"

.PHONY: generate-copilot-contexts
```

**Automation:**

Add to pre-commit hook or CI/CD:
```yaml
# .github/workflows/update_copilot_contexts.yml
name: Update Copilot Contexts
on:
  push:
    paths:
      - 'app/db/schema/hm_schema.sql'
      - 'scripts/add_pattern.sql'

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Generate contexts
        run: |
          python scripts/generate_copilot_contexts.py
      - name: Commit if changed
        run: |
          git config user.name "github-actions"
          git add .copilot/
          git commit -m "Auto-update Copilot contexts" || true
          git push
```

**Advantages Over Tag-Based Approach:**

| Tag-Based | Embedding-Based (This Script) |
|-----------|-------------------------------|
| Manual tagging required | Automatic semantic matching |
| Tags can be inconsistent | Embeddings capture meaning |
| Hard to find related patterns | Similarity scores show relevance |
| All-or-nothing (tag match) | Gradual relevance (threshold) |
| Limited to predefined tags | Works with any new pattern |

**Dependencies:**
```txt
scikit-learn>=1.3.0  # For cosine_similarity
numpy>=1.24.0
rich>=13.0.0         # For pretty console output
```

---

### Phase 4: Integration with AI Agents (Week 3)

**Goal:** Enable Aletheia and other AI agents to query patterns.

#### Task 4.1: Aletheia Context Injection

**Concept:** When Aletheia is working on code, she can query relevant patterns and inject them into her context window.

**Example Workflow:**
1. User asks: "Aletheia, add error handling to the `/chat` endpoint"
2. Aletheia queries: `POST /api/v1/context/markdown` with `query="error handling API endpoints"` and `file_pattern="api/*.py"`
3. Aletheia receives markdown with relevant patterns
4. Aletheia uses patterns to generate code that follows project conventions

**Implementation Options:**

**Option A: Manual Context Injection (Immediate)**
```python
# In Aletheia's prompt/context
"""
Before implementing, I'll check for relevant patterns:

Query: "error handling API endpoints"
File pattern: "api/*.py"

Retrieved patterns:
{markdown_from_database}

Now implementing based on these patterns...
"""
```

**Option B: Automatic Context Injection (Advanced)**
Create a middleware that automatically fetches patterns based on:
- Current file being edited
- Keywords in user's request
- Recent code changes

**File:** `app/services/ai_context_middleware.py` (new file)

```python
"""Middleware to automatically inject relevant patterns into AI context."""

from typing import List, Optional

from app.services.context_service import ContextService
from app.db.setup.database import get_db


class AIContextMiddleware:
    """Automatically enriches AI context with relevant patterns."""

    def __init__(self):
        self.db = next(get_db())
        self.context_service = ContextService(self.db)

    def enrich_context(
        self,
        user_query: str,
        current_file: Optional[str] = None,
        max_patterns: int = 3,
    ) -> str:
        """Enrich AI context with relevant patterns.
        
        Args:
            user_query: User's request to the AI
            current_file: Current file being edited (optional)
            max_patterns: Max number of patterns to retrieve
            
        Returns:
            Markdown string to prepend to AI context
        """
        # Extract file pattern from current file
        file_pattern = None
        if current_file:
            # e.g., "app/api/routes.py" → "api/*.py"
            parts = current_file.split("/")
            if len(parts) >= 2:
                file_pattern = f"{parts[-2]}/*.py"
        
        # Query patterns
        patterns = self.context_service.search_patterns(
            query=user_query,
            file_pattern=file_pattern,
            limit=max_patterns,
        )
        
        if not patterns:
            return ""
        
        # Format as markdown
        markdown = self.context_service.format_as_markdown(patterns)
        
        # Wrap in context marker
        enriched = f"""
---
## Relevant Project Patterns (Auto-Retrieved)

{markdown}

---

Now addressing user request with these patterns in mind...
"""
        return enriched


# Example usage in AI agent
def process_ai_request(user_query: str, current_file: str = None) -> str:
    middleware = AIContextMiddleware()
    
    # Get enriched context
    pattern_context = middleware.enrich_context(user_query, current_file)
    
    # Combine with user query
    full_context = f"{pattern_context}\n\nUser: {user_query}"
    
    # Send to AI model
    # response = openai.ChatCompletion.create(...)
    
    return full_context
```

#### Task 4.2: GitHub Copilot Integration (Advanced)

**Note:** GitHub Copilot (me) can't directly query your database in real-time, but we can create a **hybrid approach**:

1. **Pre-compute context files**: Generate markdown files from patterns
2. **Add to workspace**: Place in `.copilot/` or `docs/patterns/`
3. **Copilot reads**: I can read these files as part of workspace context

**Script to sync patterns to markdown files:**

**File:** `scripts/sync_patterns_to_files.py` (new file)

```python
"""Sync eng_patterns from database to markdown files for Copilot."""

from pathlib import Path

from app.db.models import EngPattern
from app.db.setup.database import get_db
from app.logging_utils import logger


def sync_patterns_to_markdown(output_dir: Path = Path("docs/patterns")):
    """Export patterns from database to markdown files.
    
    Creates one file per strategy_type, making it easy for Copilot
    to access relevant patterns based on file context.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    db = next(get_db())
    patterns = db.query(EngPattern).all()
    
    # Group by strategy_type
    by_type = {}
    for pattern in patterns:
        type_key = pattern.strategy_type or "general"
        if type_key not in by_type:
            by_type[type_key] = []
        by_type[type_key].append(pattern)
    
    # Write one file per type
    for strategy_type, patterns_list in by_type.items():
        filename = f"{strategy_type.replace(' ', '_')}.md"
        filepath = output_dir / filename
        
        md_parts = [f"# {strategy_type.replace('_', ' ').title()} Patterns\n\n"]
        md_parts.append("_Auto-generated from eng_patterns database_\n\n")
        md_parts.append("---\n\n")
        
        for pattern in patterns_list:
            md_parts.append(f"## Pattern: {pattern.id}\n\n")
            md_parts.append(f"**Tags:** {', '.join(pattern.tags or [])}\n\n")
            md_parts.append(f"**Target Contexts:** {', '.join(pattern.target_contexts or [])}\n\n")
            md_parts.append(f"{pattern.content}\n\n")
            md_parts.append("---\n\n")
        
        filepath.write_text("".join(md_parts))
        logger.info(f"✓ Wrote {len(patterns_list)} patterns to {filepath}")
    
    logger.info(f"Pattern sync complete: {len(by_type)} files written to {output_dir}")


if __name__ == "__main__":
    sync_patterns_to_markdown()
```

**Add to Makefile:**
```makefile
sync-patterns:
	@echo "Syncing patterns from database to markdown files..."
	python scripts/sync_patterns_to_files.py
```

**Usage:**
```bash
# Sync patterns to docs/patterns/
make sync-patterns

# Now Copilot can read docs/patterns/*.md
```

---

### Phase 5: Advanced Features (Week 4+)

#### Feature 5.1: Pattern Versioning
Track changes to patterns over time:

```sql
CREATE TABLE eng_pattern_versions (
    id uuid PRIMARY KEY,
    pattern_id uuid REFERENCES eng_patterns(id),
    version int NOT NULL,
    content text NOT NULL,
    changed_by text,
    changed_at timestamp with time zone DEFAULT now(),
    change_reason text
);
```

#### Feature 5.2: Pattern Usage Analytics
Track which patterns are most useful:

```sql
CREATE TABLE eng_pattern_usage (
    id uuid PRIMARY KEY,
    pattern_id uuid REFERENCES eng_patterns(id),
    used_by text,  -- 'aletheia', 'copilot', 'user_123'
    used_at timestamp with time zone DEFAULT now(),
    context jsonb  -- What was the user doing?
);
```

#### Feature 5.3: Pattern Recommendations
Suggest patterns based on:
- Current file being edited
- Recent commits
- Common mistakes

```python
def recommend_patterns(current_file: str, recent_code: str) -> List[EngPattern]:
    """Recommend patterns based on context."""
    # Extract keywords from code
    keywords = extract_keywords(recent_code)
    
    # Query patterns
    patterns = context_service.search_patterns(
        query=" ".join(keywords),
        file_pattern=infer_file_pattern(current_file),
        limit=3,
    )
    
    return patterns
```

#### Feature 5.4: Pattern Quality Scoring
Rate patterns based on:
- Usage frequency
- User feedback
- Code quality improvements

```python
class PatternQualityScorer:
    def calculate_score(self, pattern: EngPattern) -> float:
        """Calculate quality score (0-100)."""
        usage_count = get_usage_count(pattern.id)
        feedback_score = get_average_feedback(pattern.id)
        recency = get_recency_score(pattern.last_updated)
        
        return (usage_count * 0.4) + (feedback_score * 0.4) + (recency * 0.2)
```

---

## 🧪 Testing Strategy

### Unit Tests
**File:** `tests/test_context_service.py` (new file)

```python
"""Tests for context service and pattern retrieval."""

import pytest
from app.services.context_service import ContextService
from app.db.models import EngPattern


def test_search_patterns(db_session, sample_pattern):
    """Test semantic search for patterns."""
    cs = ContextService(db_session)
    
    results = cs.search_patterns("validation testing")
    assert len(results) > 0
    assert isinstance(results[0], EngPattern)


def test_search_with_file_pattern(db_session, sample_pattern):
    """Test filtering by file pattern."""
    cs = ContextService(db_session)
    
    results = cs.search_patterns("testing", file_pattern="**/*_test.py")
    assert all("test" in str(p.target_contexts) for p in results)


def test_format_as_markdown(db_session, sample_pattern):
    """Test markdown formatting."""
    cs = ContextService(db_session)
    patterns = cs.search_patterns("validation")
    
    markdown = cs.format_as_markdown(patterns)
    assert "# Engineering Patterns" in markdown
    assert "Tags:" in markdown


@pytest.fixture
def sample_pattern(db_session):
    """Create a sample pattern for testing."""
    pattern = EngPattern(
        content="# Test Pattern\n\nThis is a test.",
        tags=["testing", "sample"],
        strategy_type="test_pattern",
        target_contexts=["**/*_test.py"],
        embedding=[0.1] * 1536,  # Dummy embedding
    )
    db_session.add(pattern)
    db_session.commit()
    return pattern
```

### Integration Tests
```python
def test_api_search_endpoint(client):
    """Test /context/search endpoint."""
    response = client.post(
        "/api/v1/context/search",
        json={
            "query": "validation testing",
            "limit": 5,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "patterns" in data
    assert data["total_results"] >= 0


def test_api_markdown_endpoint(client):
    """Test /context/markdown endpoint."""
    response = client.post(
        "/api/v1/context/markdown",
        json={
            "query": "error handling",
            "file_pattern": "api/*.py",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "markdown" in data
    assert "# Engineering Patterns" in data["markdown"]
```

---

## 📝 Usage Examples

### Example 1: Aletheia Queries Patterns
```python
# Aletheia is implementing a new API endpoint
# She queries relevant patterns first

from app.services.context_service import ContextService
from app.db.setup.database import get_db

db = next(get_db())
cs = ContextService(db)

# Query patterns
patterns = cs.search_patterns(
    query="API error handling best practices",
    file_pattern="api/*.py",
    tags=["error_handling", "api"],
    limit=3,
)

# Format as markdown
context_md = cs.format_as_markdown(patterns)

# Aletheia uses this context to write better code
print(context_md)
```

### Example 2: Pre-commit Hook to Suggest Patterns
```bash
#!/bin/bash
# .git/hooks/pre-commit

# Check if modified files match any pattern contexts
python scripts/suggest_patterns_for_commit.py
```

**File:** `scripts/suggest_patterns_for_commit.py`
```python
"""Suggest relevant patterns for files in current commit."""

import subprocess
from app.services.context_service import ContextService
from app.db.setup.database import get_db

# Get modified files
result = subprocess.run(["git", "diff", "--name-only", "--cached"], capture_output=True)
files = result.stdout.decode().splitlines()

db = next(get_db())
cs = ContextService(db)

for file in files:
    # Check if we have patterns for this file
    patterns = cs.search_patterns(
        query=f"patterns for {file}",
        file_pattern=file,
        limit=1,
    )
    
    if patterns:
        print(f"\n📚 Suggested pattern for {file}:")
        print(patterns[0].content[:200] + "...")
```

### Example 3: VS Code Extension (Future)
```typescript
// .vscode/extensions/aletheia-patterns/src/extension.ts

// When user opens a file, fetch relevant patterns
vscode.workspace.onDidOpenTextDocument(async (document) => {
  const filePath = document.fileName;
  
  const response = await fetch('http://localhost:8000/api/v1/context/search', {
    method: 'POST',
    body: JSON.stringify({
      query: `patterns for ${filePath}`,
      file_pattern: inferPattern(filePath),
      limit: 3,
    }),
  });
  
  const patterns = await response.json();
  
  // Show patterns in sidebar
  patternsView.render(patterns);
});
```

---

## 🚦 Deployment Checklist

### Pre-deployment
- [ ] `eng_patterns` table exists and is accessible
- [ ] Sample patterns inserted and have embeddings
- [ ] `EngPattern` model added to `models.py`
- [ ] `ContextService` implemented and tested
- [ ] API endpoints added and tested
- [ ] Pattern management CLI tested

### Testing
- [ ] Unit tests pass for `ContextService`
- [ ] Integration tests pass for API endpoints
- [ ] Semantic search returns relevant results
- [ ] Markdown formatting is correct
- [ ] Performance is acceptable (< 200ms for searches)

### Production
- [ ] Run `make sync-patterns` to export patterns for Copilot
- [ ] Document pattern addition workflow in `CONTRIBUTING.md`
- [ ] Set up periodic pattern sync (cron job or GitHub Action)
- [ ] Monitor API endpoint performance
- [ ] Collect feedback on pattern usefulness

---

## 📊 Success Metrics

1. **Pattern Usage**
   - Number of patterns retrieved per day
   - Most frequently retrieved patterns
   - Patterns with zero usage (candidates for removal)

2. **Code Quality**
   - Reduction in code review comments on style/patterns
   - Increase in test coverage
   - Decrease in bugs related to pattern violations

3. **Developer Experience**
   - Time saved by using patterns vs searching documentation
   - Developer satisfaction with pattern system
   - Number of new patterns contributed by team

4. **AI Effectiveness**
   - Aletheia's code quality when using patterns vs not
   - Pattern retrieval accuracy (relevant vs irrelevant)
   - Context window efficiency (tokens used for patterns)

---

## 🎓 Pattern Authoring Guidelines

When creating new patterns for `eng_patterns`:

### Structure
```markdown
# Pattern Title

Brief description of the pattern.

## When to Use
- Specific scenarios where this pattern applies

## Implementation
```python
# Code example
def example():
    pass
```

## Benefits
1. Clear benefit 1
2. Clear benefit 2

## Anti-Patterns to Avoid
- What NOT to do
```

### Metadata
- **Tags**: Use consistent, searchable tags (e.g., `testing`, `error_handling`, `python`)
- **Target contexts**: Be specific with glob patterns (e.g., `api/*.py`, `tests/**/*_test.py`)
- **Strategy type**: Choose from: `design_pattern`, `coding_standard`, `integration_guide`, `troubleshooting`, `architecture`

### Quality Checklist
- [ ] Clear title and description
- [ ] Code examples included
- [ ] When to use vs when NOT to use
- [ ] Tags are consistent with existing patterns
- [ ] Target contexts accurately match relevant files
- [ ] Content is searchable (good keywords)

---

## 🔗 Related Documentation

- **Schema:** `app/db/schema/hm_schema.sql` - Database schema with example pattern
- **Models:** `app/db/models.py` - SQLAlchemy ORM model (to be added)
- **Embeddings:** `app/utils/embeddings.py` - Embedding generation utilities
- **API:** `app/api/routes.py` - API endpoints (to be added)

---

## 🎯 Next Actions

1. **Immediate (Week 1)**
   - [ ] Add `EngPattern` model to `models.py`
   - [ ] Implement `ContextService` with semantic search
   - [ ] Add basic API endpoints (`/context/search`, `/context/markdown`)
   - [ ] Write unit tests

2. **Short-term (Week 2-3)**
   - [ ] Create pattern management CLI
   - [ ] Build pattern import/export tools
   - [ ] Sync patterns to markdown files for Copilot
   - [ ] Add more example patterns to database

3. **Medium-term (Week 4+)**
   - [ ] Implement AI context middleware
   - [ ] Add pattern usage analytics
   - [ ] Create VS Code extension (optional)
   - [ ] Build pattern recommendation system

4. **Long-term (Month 2+)**
   - [ ] Pattern versioning system
   - [ ] Quality scoring and feedback loop
   - [ ] Multi-language support (TypeScript, Rust, etc.)
   - [ ] Pattern generation from existing codebase

---

## 💡 Pro Tips

1. **Start Small**: Begin with 5-10 high-quality patterns covering your most common tasks
2. **Iterate**: Get feedback from Aletheia and team on pattern usefulness
3. **Keep Current**: Update patterns as codebase conventions evolve
4. **Measure Impact**: Track before/after code quality metrics
5. **Automate**: Set up CI/CD to sync patterns regularly

---

**Questions or issues?** Contact James or file an issue with tag `documentation-system`.
