# 📊 01 · Data Pipeline — Foundation of Predictive Intelligence

> _“No AI model can outlearn the flaws of its data. In predictive betting, information precision defines profitability.”_

This module expands **Section 1 · High-Quality Data Curation** of the main `README` and gives implementation-level detail.

---

## 1  Overview

The VEGAIS data pipeline builds a **low-latency knowledge graph** of sports events, odds flows and social signals.  
Let \( \mathcal{D} = \{(x_i, y_i)\} \) denote the curated dataset used to learn \( P_\theta(y \mid x) \).

---

## 2  Sources of Truth <a id="sources"></a>

| Data Class | Description | Primary Feed | Update Freq |
|------------|-------------|--------------|-------------|
| Match Metadata | league, kickoff, rosters | Sportradar API | 1 min |
| Live Stats | goals, xG, fouls, injuries | StatsPerform | 5 s |
| Betting Odds | pre-match & in-play lines | 25+ books (Bet365, Pinnacle) | ≤2 s |
| Social/News | lineup leaks, injuries | Twitter fire-hose, RSS | 30 s |
| On-chain Bets | wallet-level wagers | TON GraphQL | near-block |

*The **Betting Odds** feed is essential for ROI benchmarks listed in README §4.*

---

## 3  Modular Architecture

```mermaid
flowchart TD
    A[Raw Feeds] --> B[Parser & Struct-VBERT Schema]  %% match README wording
    B --> C{Validator}
    C -->|✔︎| D[Feature Store¹]
    C -->|✖︎| E[Noise Queue]
    D --> F[Version Registry²]
    F --> G[Trainer Queue]
```

> ¹ *Feature Store aligns with the “Feature Store” node in the main README diagram.*  
> ² *Snapshot-hash protocol (§5) fulfils the “Every change hashed & tagged” guarantee in README §1.3.*

---

## 4  Annotation & Human–AI Loop

We apply semi-supervised consensus:  
\[
L = \alpha\,L_{AI} + (1-\alpha)\,L_{Human},\quad \alpha = 0.6
\]

Resulting label accuracy ≥ 96 % on hold-out (*matches README evaluation κ > 0.83*).

---

## 5  Governance & Versioning

All transformations emit SHA-256 hashes (DVC) → immutable lineage.  
Each model in README §3 references a snapshot ID \( S_t \).

---

## 6  External Benchmarks

We align our pipeline outputs with public prediction datasets:
- StatsBomb xG Open Set [2]  
- Kaggle Football Results [3]  
- NFL Big Data Bowl [4]

Evaluation metrics (LogLoss, Brier) match README §4.

---

## 7  Engineering Stack

| Layer | Tooling | Notes |
|-------|---------|-------|
| Ingest | **Puppeteer**, **Playwright**, Scrapy | handles JS-heavy books |
| Stream | WebSockets + Kafka | 2 s end-to-end |
| Transform | Pandas / Polars | SIMD-optimised |
| Store | PostgreSQL + Weaviate | hybrid queries |
| Orchestrate | Airflow / Prefect | DAG versioning |
| Version | DVC, LakeFS | snapshot-hash |
| Monitor | Prometheus, Grafana | SLO alerts |

---

## 8  DevOps SLA

- Mean ingestion latency \(< 3\,\text{s}\)  
- Pipeline uptime 99.98 % (2025 Q1)  
- Canary tests track drift vs README ROI metrics.

---

## References

[1] Anderson et al., “Data-Centric AI”, NeurIPS 2021  
[2] StatsBomb Open xG, https://github.com/statsbomb/open-data  
[3] Kirkaal, Football Results 2010–2023, Kaggle  
[4] NFL Big Data Bowl, Kaggle 2021  

---

> **Note:** This pipeline module is fully consistent with *High-Quality Data Curation* in the main **AI Prediction Model Research** README.


---

## Appendix A · Principles and Process of High-Quality Data Construction

Based on industry-standard practices and internal frameworks, VEGAIS defines high-quality data construction as a lifecycle process:

### A.1 Core Questions

- What are the fundamental principles of high-quality dataset construction?
- How should continuous training and fine-tuning datasets be structured in different industry contexts?
- Are there standardized pipelines and toolkits available to support this process?

### A.2 Construction Workflow

| Stage          | Sub-processes                                  |
|----------------|-------------------------------------------------|
| **Pipeline**   | Data Collection · Data Preprocessing · Data Annotation · Data Augmentation · Data Evaluation · Iteration & Optimization |
| **Build Phase**| Data Governance · Dynamic Data Updates         |
| **Core Principles** | Quality · Scale · Relevance · Diversity · Timeliness · Updateability |

This section reflects the internal lifecycle from raw ingestion to structured predictive training sets and continuous refreshment.

> _Diagram source: VEGAIS Lab Internal Toolchain Planning v1.1, 2025_


---

## Appendix B · Sports Match Data Schema (Sample)

To support rich prediction tasks and betting strategy modeling, we define an extensible schema of sports match data categories and fields:

### B.1 Core Categories and Fields

| Category          | Field Name                  | Description |
|------------------|-----------------------------|-------------|
| Match Info        | match_id                    | Unique game identifier |
|                  | league_name                 | League or tournament |
|                  | season                      | Season or split |
|                  | kickoff_time                | UTC timestamp of start |
| Team Info         | home_team                   | Name of home team |
|                  | away_team                   | Name of away team |
|                  | home_team_rating            | ELO / power score |
|                  | away_team_rating            | ELO / power score |
| Player Info       | lineup_home_players         | List of player IDs |
|                  | lineup_away_players         | List of player IDs |
|                  | injuries_pre_match          | Known injury flags |
| Score Event       | score_home                  | Final score home |
|                  | score_away                  | Final score away |
|                  | halftime_score              | Score at half-time |
| Match Stats       | shots_on_target_home        | Quantitative metrics |
|                  | shots_on_target_away        |                   |
|                  | xG_home                     | Expected Goals (home) |
|                  | xG_away                     | Expected Goals (away) |
| Betting Markets   | odds_home_win               | Decimal odds |
|                  | odds_draw                   |                  |
|                  | odds_away_win               |                  |

### B.2 Notes

- All fields are time-indexed and versioned per snapshot (see §5)  
- Compatible with public schemas such as [StatsBomb](https://github.com/statsbomb/open-data) and [Opta API]

---

> _Schema is extensible to sports such as basketball, tennis, esports, or racing. Used across both model training and bet execution engines._
