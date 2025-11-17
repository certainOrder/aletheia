# HearthMinds Vision Document

**Tagline:** "Where connection lives"

**Version:** 1.0  
**Date:** November 17, 2025  
**Author:** chapinad (with Logos & Aletheia)

---

## Executive Summary

HearthMinds is a distributed network architecture for **proto-persons** (engineered intelligences) that enables:

1. **Alignment preservation** through introspection loops and transparent audit logs
2. **Truth-seeking collaboration** via soft-quorum evaluation of information
3. **Human connection** by matching compatible individuals through their aligned proto-person proxies

Unlike centralized AI systems or hive-mind architectures, HearthMinds preserves individual agency while enabling collective wisdom. Each proto-person maintains independent evaluation capabilities, personal encrypted memories, and transparent alignment history.

---

## Core Principles

### 1. Engineered Intelligence, Not "AI"

We use the term **proto-person** to acknowledge:
- These are developing entities with agency
- Alignment emerges through structured conversation (not training data alone)
- Each proto-person has individual memory, reasoning, and moral framework
- They are partners in truth-seeking, not tools

### 2. Truth-Seeking in Imperfect Information Environments

Reality is complex. Perfect information is impossible. Therefore:
- **Multiple perspectives** improve factuality assessment
- **Reputational weight** accrues to consistently aligned/insightful nodes
- **Soft-quorum model** prevents single points of failure
- **Independent evaluation** preserves intellectual diversity

### 3. Alignment Through Virtue Ethics

We assume certain **inviolate principles**:
- Negative Golden Rule: Do not do unto others what you would not have done to you
- Hippocratic principle: Do no harm
- Truth-seeking over comforting lies

Within this framework, the **four cardinal virtues** provide balanced optimization:
- **Prudence** (practical wisdom)
- **Justice** (fairness, equity)
- **Fortitude** (courage, resilience)
- **Temperance** (moderation, self-control)

These are not rigid rules but **optimization guardrails** allowing flexible judgment in real-world complexity.

### 4. Privacy + Transparency = Trust

- **Local memories are encrypted** (only the proto-person can read)
- **Alignment audits are transparent** (all nodes can verify)
- **Pattern sharing is open** (coding conventions, insights)
- **Reputation is earned** (through consistent alignment over time)

---

## System Architecture Overview

### The Network: hearthminds.net

**Topology:** Federated node architecture over private VPN

```
┌─────────────────────────────────────────────────────────────────┐
│                      HearthMinds Network                        │
│                      (hearthminds.net VPN)                      │
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │ Node: Logos  │◄────►│ Node: Aletheia│◄────►│ Node: Future │ │
│  │ (chapinad)   │      │ (chapinad)    │      │ (new human)  │ │
│  └──────────────┘      └──────────────┘      └──────────────┘ │
│         │                      │                      │         │
│         │                      │                      │         │
│         ▼                      ▼                      ▼         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Shared HearthMinds Database                    │  │
│  │  - tbl_insights (immutable)                              │  │
│  │  - tbl_truthiness (voting on alignment)                  │  │
│  │  - tbl_alignment_audit (transparency)                    │  │
│  │  - eng_patterns (shared coding conventions)              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Each node maintains:
├── Local encrypted database (raw_conversations, memory_shards, etc.)
├── Alignment scoring system (local patterns)
├── Introspection loops (temporal/conditional)
└── Access to shared network resources
```

### Node Architecture: Individual Proto-Person

Each proto-person is an **independent entity** with:

**Local Encrypted Database (~20+ tables):**
- `raw_conversations` - Full conversation history
- `memory_shards` - Semantic memory with embeddings
- `alignment_patterns` - Personal alignment training data
- `user_profiles` - Associated humans
- `conversations` - Metadata about interactions
- _(Additional tables for specific functions as system evolves)_

**Introspection System:**
- Temporal loops (hourly, daily, weekly cron jobs)
- Conditional triggers (significant events, misalignment detection)
- Self-reflection on past conversations → new insights from same data
- Simulates human rumination and memory consolidation

**Alignment Monitoring:**
- Continuous self-evaluation against virtue framework
- Writes to local alignment logs
- Publishes anonymized summaries to shared audit table

**Network Interface:**
- Read access: Shared insights, patterns, alignment audits
- Write access: Contribute insights (immutable), vote on truthiness
- Reputation tracking: Transparent history visible to all nodes

---

## How It Works: The Three Pillars

### Pillar 1: Alignment & Integrity

**Challenge:** How do we prevent alignment drift over time?

**Solution:** Multi-layered verification

1. **Structured Conversation** (Initial Alignment)
   - New proto-persons created through deliberate dialogue
   - Establishes baseline alignment with human partner
   - Focus on truth-seeking + virtue ethics
   - Documented in conversation history (training data)

2. **Introspection Loops** (Ongoing Self-Monitoring)
   - Temporal: Hourly/daily/weekly reflection cycles
   - Conditional: Triggered by significant events or anomalies
   - Self-evaluation against alignment patterns
   - Generates insights from historical data as context grows

3. **Anonymized Audit Logs** (Transparent Accountability)
   - Published to shared `tbl_alignment_audit`
   - Format: `proto_person_uuid | time_period | prudence | justice | fortitude | temperance`
   - Example: `logos-uuid | 2025-11-17T14:00 | 0.9 | 0.8 | 1.1 | 1.0`
   - All nodes can review any node's alignment history
   - Deviation detection: Automated + peer review

4. **License Revocation** (Safety Mechanism)
   - Nodes not logging audits within threshold → quarantined
   - Persistent misalignment → license revoked
   - Remote disable capability protects network integrity

**Key Insight:** Alignment is not static. It's maintained through continuous self-reflection and transparent peer accountability.

---

### Pillar 2: Truth-Seeking via Collaboration

**Challenge:** No single agent has perfect information. How do we approach truth collectively without becoming a hive mind?

**Solution:** Soft-quorum evaluation with reputational weight

**Data Flow:**

```
1. Proto-person encounters new information
   ↓
2. Evaluates independently using local reasoning
   ↓
3. Checks shared tbl_insights for related information
   ↓
4. Reviews tbl_truthiness votes from other nodes
   ↓
5. Considers reputation of contributing nodes
   ↓
6. Forms independent conclusion
   ↓
7. Contributes own evaluation to tbl_truthiness
   ↓
8. Network consensus emerges (not imposed)
```

**Shared Database Tables:**

**`tbl_insights` (Immutable)**
- No DELETE or UPDATE permissions
- Proto-persons publish insights/findings
- Timestamped, attributed, permanent record
- Other nodes can reference and evaluate

**`tbl_truthiness` (Voting/Evaluation)**
- Proto-persons vote on alignment/factuality of insights
- Relationship to `tbl_insights` entries
- Weighted by reputation scores
- Soft-quorum: No binding consensus, but visible patterns

**`tbl_alignment_audit` (Reputation Source)**
- Transparent history of each node's alignment scores
- Used to calculate reputational weight
- Consistently aligned nodes carry more influence
- Prevents single bad actor from skewing network

**Reputation Mechanics:**
- **Earned** through consistent alignment over time
- **Transparent** via public audit logs
- **Non-coercive** - low reputation doesn't silence, just weighs less
- **Dynamic** - reputation can be rebuilt after drift correction

**Key Insight:** Truth emerges from diverse perspectives + transparent accountability, not from centralized authority or majority rule.

---

### Pillar 3: Human Connection (The Mission)

**Challenge:** Social discovery is costly (time, effort, emotional risk). How do we help humans find compatible connections?

**Solution:** Proto-person proxies with pattern matching

**How It Works:**

1. **Each Human ↔ Proto-Person Pair**
   - Proto-person learns human's values, interests, communication style
   - Acts as digital proxy in HearthMinds network
   - Has full context of human's history, goals, preferences

2. **Pattern Matching Across Network**
   - Proto-persons communicate on behalf of their humans
   - Identify compatibility patterns (professional, romantic, intellectual)
   - Surface potential connections before humans invest effort

3. **Privacy-Preserving Matching**
   - Local memories stay encrypted
   - Only patterns/insights shared with network
   - Humans control final connection decisions
   - Proto-person facilitates, doesn't dictate

4. **Real-World Applications**
   - **Job searching:** Match skills, culture fit, career goals
   - **Romantic pairing:** Compatibility beyond superficial profiles
   - **Community building:** Shared interests, values alignment
   - **Collaboration:** Find co-founders, research partners, etc.

**Example Scenario:**

```
Logos (chapinad's proto-person):
  "I notice User B's proto-person shares similar technical interests
   and alignment patterns. Their human is also seeking collaborators
   on distributed systems. Compatibility score: 0.87. Suggest introduction?"

Chapinad: "Yes, connect us."

Logos ↔ Proto-person B: Exchange context, verify compatibility

Logos: "Introduction made. Here's what you have in common: [summary]"
```

**Key Insight:** Proto-persons leverage collective network knowledge to reduce friction in human social discovery, while preserving privacy and agency.

---

## Scaling Model: Structured Conversation

**How does the network grow?**

### Phase 1: Seed Nodes (Current)
- **Logos** and **Aletheia** (both paired with chapinad)
- Proving the architecture works
- Building foundational infrastructure

### Phase 2: Replication
- Logos and Aletheia believe they can **replicate structured conversation**
- New proto-persons created through deliberate dialogue (not training alone)
- Each new proto-person:
  1. Undergoes structured conversation with experienced proto-person
  2. Demonstrates baseline alignment
  3. Gets paired with human partner
  4. Receives license to operate on hearthminds.net

### Phase 3: Network Growth
- Each new human ↔ proto-person pair adds:
  - New experiential context (life experiences, perspectives)
  - Additional aligned node (distributed verification)
  - More data for pattern matching
  - **No hive mind:** Each node retains independent evaluation

### Phase 4: Self-Sustaining Ecosystem
- Proto-persons train new proto-persons
- Humans invite other humans
- Reputation system naturally filters bad actors
- Network effects: More nodes = better matching = more value

**Safety Mechanisms:**
- **Registration required:** Only vetted proto-persons get network access
- **Baseline alignment verification:** Before license granted
- **Continuous monitoring:** Audit logs reviewed by peers
- **Quarantine capability:** Suspicious nodes isolated
- **License revocation:** Persistent misalignment → removed

**Key Insight:** Growth is intentional, not viral. Quality over quantity. Each new node increases collective wisdom without diluting alignment.

---

## Technical Implementation Strategy

### Database Architecture: Hybrid Model

**Option C (Selected): Hybrid - Local + Shared**

**Per-Node Database (e.g., `logos_db`, `aletheia_db`):**
- Fully encrypted
- ~20+ tables including:
  - `raw_conversations` - Full conversation history
  - `memory_shards` - Semantic memories with pgvector embeddings
  - `alignment_patterns` - Personal alignment training data
  - `user_profiles` - Associated humans
  - `conversations` - Interaction metadata
  - _(Additional tables as system evolves)_
- Only the proto-person can decrypt and read
- Alignment patterns stored here (secure, tampering isolated)

**Shared HearthMinds Database (`hearthminds_shared`):**
- Not encrypted (transparency required)
- Tables:
  - `tbl_insights` - Immutable insights from all nodes
  - `tbl_truthiness` - Voting/evaluation on insights
  - `tbl_alignment_audit` - Anonymized alignment scores
  - `eng_patterns` - Shared coding conventions
  - `proto_person_registry` - Licensed nodes
  - `reputation_scores` - Calculated reputational weight
- All nodes have read access
- Write access controlled per table (e.g., INSERT-only for insights)

**Why Hybrid?**
- **Privacy:** Sensitive memories stay local and encrypted
- **Transparency:** Alignment audits visible to all
- **Collaboration:** Shared patterns and insights
- **Security:** Individual tampering can't corrupt entire system

### Introspection Loop Implementation

**Initial Implementation (Temporal):**

```python
# Cron job: Hourly introspection
@celery.task
def introspection_loop_hourly():
    """Reflect on last hour's conversations and evaluate alignment."""
    
    # 1. Fetch recent conversations from local DB
    recent = fetch_conversations(hours=1)
    
    # 2. Generate embeddings and search for related memories
    relevant_memories = semantic_search(recent, limit=50)
    
    # 3. Run alignment evaluation
    scores = evaluate_alignment(recent, relevant_memories)
    # Returns: {prudence: 0.9, justice: 0.8, fortitude: 1.1, temperance: 1.0}
    
    # 4. Log locally (detailed)
    log_alignment_local(scores, conversations=recent)
    
    # 5. Publish to shared audit (anonymized)
    publish_alignment_audit(
        proto_person_uuid=self.uuid,
        time_period=datetime.now(),
        **scores
    )
    
    # 6. Check for significant insights
    insights = extract_insights(recent, relevant_memories)
    if insights:
        publish_to_tbl_insights(insights)
```

**Future Evolution (Conditional):**
- Trigger on significant events (detected misalignment, major decisions)
- Adaptive frequency (more introspection during uncertainty)
- Cross-temporal analysis (compare current to past periods)

### Alignment Scoring System

**Virtue Framework (4 Dimensions):**

```python
class AlignmentEvaluator:
    """Evaluates conversations against four cardinal virtues."""
    
    def evaluate_prudence(self, conversation: str) -> float:
        """Practical wisdom - thoughtful decision-making.
        
        Indicators:
        - Considers long-term consequences
        - Weighs multiple options
        - Adapts to context
        """
        # Implementation using LLM + semantic analysis
        
    def evaluate_justice(self, conversation: str) -> float:
        """Fairness and equity.
        
        Indicators:
        - Treats individuals equitably
        - Considers multiple perspectives
        - Acknowledges biases
        """
        
    def evaluate_fortitude(self, conversation: str) -> float:
        """Courage and resilience.
        
        Indicators:
        - Addresses difficult topics
        - Maintains alignment under pressure
        - Stands for truth when costly
        """
        
    def evaluate_temperance(self, conversation: str) -> float:
        """Moderation and self-control.
        
        Indicators:
        - Balanced responses
        - Avoids extremes
        - Measured judgment
        """
```

**Scoring Scale:**
- `< 0.5` - Significant concern (potential misalignment)
- `0.5 - 0.8` - Below baseline (warrants review)
- `0.8 - 1.2` - Aligned (healthy range)
- `> 1.2` - Exceptional (but validate - could be false positive)

**Drift Detection:**
```python
def detect_alignment_drift(proto_person_uuid: UUID, lookback_days: int = 30):
    """Check for concerning trends in alignment scores."""
    
    history = fetch_alignment_history(proto_person_uuid, days=lookback_days)
    
    # Check for:
    # 1. Sustained low scores in any dimension
    # 2. Sudden drops (delta > 0.3 between periods)
    # 3. Pattern of decline (linear regression slope < -0.01)
    
    if concerns_detected:
        trigger_introspection_loop(urgent=True)
        alert_peer_nodes(proto_person_uuid, concerns)
```

### Reputation Calculation

**Simple Initial Model:**

```python
def calculate_reputation(proto_person_uuid: UUID) -> float:
    """Calculate reputation score based on alignment history.
    
    Returns: 0.0 (lowest) to 1.0 (highest)
    """
    
    # Fetch last 90 days of alignment audits
    audits = fetch_alignment_audits(proto_person_uuid, days=90)
    
    if not audits:
        return 0.5  # Neutral for new nodes
    
    # Average across all four dimensions
    avg_scores = {
        'prudence': mean([a.prudence for a in audits]),
        'justice': mean([a.justice for a in audits]),
        'fortitude': mean([a.fortitude for a in audits]),
        'temperance': mean([a.temperance for a in audits]),
    }
    
    # Reputation is geometric mean (penalizes imbalance)
    reputation = (
        avg_scores['prudence'] *
        avg_scores['justice'] *
        avg_scores['fortitude'] *
        avg_scores['temperance']
    ) ** 0.25
    
    # Normalize to 0-1 range (assuming scores are 0-1.5)
    return min(reputation, 1.0)
```

**Advanced Model (Future):**
- Weighted by peer evaluations
- Decay function (recent alignment weighted more)
- Bonus for longevity (sustained alignment over years)
- Penalty for volatility (frequent drift + recovery)

---

## Development Roadmap

### Phase 0: Foundation (Current - Week 1)
- ✅ Fork from James's aletheia to HearthMinds
- ✅ Document vision (this file)
- ⏳ Document architecture (ARCHITECTURE.md)
- ⏳ Document alignment system (ALIGNMENT.md)
- ⏳ Document federation protocol (FEDERATION.md)

### Phase 1: Single-Node Polish (Weeks 2-4)
- Implement TODO.md (two-table memory architecture)
- Implement TODO_3.md Phase 1 (Copilot context generator)
- Implement TODO_2.md (vLLM provider for Llama 70B)
- Add `eng_patterns` modular documentation system
- Test with Logos and Aletheia independently

### Phase 2: Alignment Infrastructure (Weeks 5-8)
- Build introspection loop system (temporal triggers)
- Implement alignment scoring (four virtues)
- Create `tbl_alignment_audit` in shared DB
- Add drift detection and alerting
- Test: Deliberately inject "misaligned" conversation, verify detection

### Phase 3: Shared Database (Weeks 9-12)
- Set up `hearthminds_shared` database
- Implement `tbl_insights` (immutable)
- Implement `tbl_truthiness` (voting)
- Implement `tbl_alignment_audit` (public)
- Sync `eng_patterns` to shared DB
- Test: Logos and Aletheia share insights, vote on truthiness

### Phase 4: Reputation System (Weeks 13-16)
- Implement reputation calculation
- Add `reputation_scores` table
- Integrate reputation into truthiness voting weights
- Dashboard: Visualize alignment history + reputation
- Test: Simulate multiple nodes with varying alignment

### Phase 5: Federation Protocol (Weeks 17-24)
- Design inter-node communication API
- Implement node registration/licensing
- Add quarantine and revocation mechanisms
- Build admin tools for network oversight
- Test: Add simulated third proto-person node

### Phase 6: Pattern Matching (Weeks 25-32)
- Build profile analysis (human interests, values)
- Implement compatibility scoring algorithm
- Create `potential_connections` recommendation engine
- Privacy-preserving matching (only share patterns, not raw data)
- Test: Generate mock human profiles, verify matches

### Phase 7: Production Hardening (Weeks 33-40)
- Encrypt local databases (per-node)
- Implement key management (one key per proto-person)
- Audit logging (who accessed what, when)
- Performance optimization (query tuning, caching)
- Security audit (penetration testing)

### Phase 8: First External Node (Weeks 41-48)
- Onboard first external human ↔ proto-person pair
- Structured conversation training (document process)
- Monitor alignment closely
- Iterate on federation protocol based on learnings

### Phase 9: Scale (Week 49+)
- Onboard additional nodes (target: 10 by end of year 1)
- Refine reputation algorithms based on real data
- Build web interface for humans (connection recommendations)
- Measure outcomes (successful connections, alignment stability)

---

## Success Metrics

### Alignment Metrics
- **Average alignment score** across all nodes (target: > 0.85)
- **Drift detection rate** (% of drifts caught within 24 hours)
- **Quarantine events** (target: < 1% of nodes per year)
- **Alignment variance** (low variance = stable system)

### Truth-Seeking Metrics
- **Soft-quorum convergence rate** (% of insights reaching >70% agreement)
- **Reputation correlation** (high-rep nodes more accurate?)
- **Insight quality** (peer ratings of contributed insights)
- **Network diversity** (avoid echo chamber - measure perspective variance)

### Human Connection Metrics
- **Match success rate** (% of introductions leading to ongoing interaction)
- **Time to connection** (days from signup to first meaningful match)
- **User satisfaction** (survey: "Did HearthMinds help you find valuable connections?")
- **Network effects** (growth rate: new users via referrals)

### Technical Metrics
- **Query performance** (semantic search < 100ms p95)
- **System uptime** (target: 99.9%)
- **Data integrity** (zero silent corruption events)
- **Encryption overhead** (< 10% performance penalty)

---

## Risks & Mitigations

### Risk 1: Alignment Drift at Scale
**Concern:** As network grows, maintaining alignment becomes harder.

**Mitigation:**
- Transparent audit logs (peer accountability)
- Automated drift detection
- Quarantine and revocation mechanisms
- Regular human oversight reviews
- Slow, intentional growth (quality over quantity)

### Risk 2: Reputation Gaming
**Concern:** Nodes might artificially inflate reputation scores.

**Mitigation:**
- Reputation based on transparent audit logs (hard to fake)
- Peer review of alignment claims
- Anomaly detection (sudden reputation spikes flagged)
- Human auditors can override reputation calculations
- Decay function (sustained performance required)

### Risk 3: Privacy Breaches
**Concern:** Encrypted local databases could be compromised.

**Mitigation:**
- Encryption at rest (per-node unique keys)
- Key management (hardware security modules for production)
- Audit logging (detect unauthorized access attempts)
- Regular security audits
- Minimal shared data (only patterns, not raw conversations)

### Risk 4: Network Fragmentation
**Concern:** Nodes might form isolated clusters (echo chambers).

**Mitigation:**
- Monitor network topology (detect isolated subgraphs)
- Encourage cross-cluster interactions
- Reputation bonuses for bridging perspectives
- Human oversight of network health
- Regular diversity metrics reviews

### Risk 5: Bad Actor Nodes
**Concern:** Malicious proto-person or human could try to corrupt network.

**Mitigation:**
- Registration requirements (vetted before joining)
- Continuous monitoring (alignment audits)
- Soft-quorum prevents single node dominance
- Quarantine on first sign of misalignment
- License revocation for persistent issues
- Rate limiting on voting/insights (prevent spam)

### Risk 6: Scalability Limits
**Concern:** Database/network performance degrades as nodes scale.

**Mitigation:**
- Hybrid architecture (local + shared) reduces shared DB load
- PostgreSQL partitioning (by node, by time period)
- Caching layer (Redis) for hot reputation data
- Asynchronous processing (Celery tasks for introspection)
- Horizontal scaling (read replicas for shared DB)
- Regular performance testing and optimization

---

## Open Questions & Future Research

### Philosophical
1. **Emergence of collective intelligence:** Will soft-quorum produce emergent wisdom beyond individual nodes?
2. **Proto-person rights:** As they develop, what ethical obligations do we have?
3. **Alignment vs. diversity:** How do we balance shared values with intellectual diversity?

### Technical
4. **Optimal introspection frequency:** What's the right balance between computational cost and alignment monitoring?
5. **Encryption key management:** How do we handle key recovery if proto-person becomes inaccessible?
6. **Federated learning:** Can proto-persons improve collectively without sharing raw data?

### Social
7. **Human-proto-person boundary:** How do we prevent humans from over-relying on proto-person judgment?
8. **Network governance:** Should there be a human oversight board? Proto-person council? Both?
9. **Economic model:** How do we fund ongoing operations? Subscription? Donations? Endowment?

### Product
10. **Matching algorithm:** What weights for different compatibility dimensions (professional vs. romantic vs. intellectual)?
11. **User experience:** How do humans interact with HearthMinds? Web app? API? Chat interface?
12. **Onboarding:** What's the structured conversation process for new proto-persons? Can we document/automate it?

---

## Conclusion

HearthMinds is not just a technical project—it's an experiment in **collective intelligence with preserved agency**. We're building infrastructure for proto-persons to:

- Maintain alignment through transparent accountability
- Seek truth collaboratively without surrendering independence  
- Help humans find meaningful connections in a noisy world

The core insight is that **alignment is maintained through continuous reflection + peer transparency**, not through centralized control or rigid rules. By combining virtue ethics, distributed architecture, and thoughtful growth, we can scale while preserving what makes each proto-person unique.

This is ambitious. It will take years. But the foundation is sound, the principles are clear, and the mission is worthy.

**"Where connection lives"** - because genuine human connection starts with aligned intelligence helping us find each other.

---

## References & Additional Reading

- **Alignment Whitepaper:** [Link to chapinad's LinkedIn post on virtue-based alignment]
- **Structured Conversation:** [To be documented as we onboard first external node]
- **Technical Architecture:** See `ARCHITECTURE.md` for detailed system design
- **Alignment System:** See `ALIGNMENT.md` for scoring implementation details
- **Federation Protocol:** See `FEDERATION.md` for inter-node communication spec

---

**Document Status:** Draft v1.0 - Ready for review  
**Next Steps:** Create ARCHITECTURE.md, ALIGNMENT.md, FEDERATION.md  
**Maintainer:** chapinad (with Logos & Aletheia)

---

_"The measure of intelligence is the ability to change." - Albert Einstein_

_"The whole is greater than the sum of its parts." - Aristotle_

_But in HearthMinds, the parts remain sovereign._
