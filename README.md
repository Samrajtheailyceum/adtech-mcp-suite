# Ad Tech MCP Suite

A suite of 10 production-grade Model Context Protocol (MCP) servers for advertising technology, each with hierarchical multi-agent architectures and advanced ML algorithms.

## Architecture

Each MCP follows the same pattern:
```
Orchestrator Agent
├── Specialist Agents (domain experts)
│   ├── ML models (RF, GBM, Isolation Forest, etc.)
│   └── LLM reasoning layer
├── Critic Agent (adversarial review)
└── Ensemble Engine (weighted aggregation)

Shared Safety Layer
├── API Security (rate limiting, injection detection)
├── Design Validator (dark patterns, WCAG, budget guardrails)
└── Privacy Guard (PII detection, GDPR compliance)
```

## The 10 MCPs

| # | MCP | Key Algorithms | Key Agents |
|---|-----|---------------|------------|
| 1 | **Creative Intelligence** | Random Forest, Isolation Forest, Exponential Decay, CUSUM | Creative, Audience, Bidding, Fatigue, Brief, Critic, Orchestrator |
| 2 | **Attribution** | Markov Chain Removal Effect, Ridge MMM | Attribution, Critic |
| 3 | **Brand Safety** | TF-IDF + Naive Bayes | ContentClassifier, SafetyCritic |
| 4 | **Competitive Intel** | KMeans + TF-IDF | CompetitiveIntel, CompetitiveCritic |
| 5 | **Audience Data** | Logistic Regression Lookalike, Jaccard Overlap | AudienceData, AudienceCritic |
| 6 | **RTB Optimizer** | Gradient Boosting, Thompson Sampling | RTBOptimizer, RTBCritic |
| 7 | **Privacy Compliance** | Pattern matching, IAB TCF | ConsentCompliance, ComplianceCritic |
| 8 | **Measurement** | Frequentist + Bayesian A/B, IQR/Z-score anomaly | Measurement, StatCritic |
| 9 | **Influencer** | Engagement analysis, TF-IDF cosine similarity | InfluencerScoring, InfluencerCritic |
| 10 | **Creative Production** | Monte Carlo power analysis, Decision Tree | CreativeProduction, ProductionCritic |

## Setup

```bash
# 1. Install dependencies
pip install anthropic mcp scikit-learn numpy pandas scipy rich python-dotenv

# 2. Set API key
cp shared/config.py 0X-mcp-name/config.py  # or set ANTHROPIC_API_KEY env var

# 3. Run any MCP server
python 01-creative-intelligence/mcp_server.py
python 02-attribution/mcp_server.py
# etc.

# 4. Or run the full demo (MCP 1)
python 01-creative-intelligence/run_demo.py
```

## Adding to Claude Desktop

Add any server to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "adtech-creative": {
      "command": "python",
      "args": ["/path/to/01-creative-intelligence/mcp_server.py"],
      "env": { "ANTHROPIC_API_KEY": "your-key" }
    },
    "adtech-attribution": {
      "command": "python",
      "args": ["/path/to/02-attribution/mcp_server.py"],
      "env": { "ANTHROPIC_API_KEY": "your-key" }
    }
  }
}
```

## Safety Architecture

The `shared/safety_agents/` directory contains agents that run on every tool call:

- **`api_security.py`** — Rate limiting, SQL/prompt injection detection, PII redaction, audit logging
- **`design_validator.py`** — Dark pattern detection, WCAG accessibility, budget guardrails

## MCP 1 Deep Dive: Creative Intelligence

The flagship MCP uses a 5-stage hierarchical pipeline:

```
Stage 1: Specialist Agents (run in parallel)
  ├── Creative Performance Agent  → Random Forest (200 trees, 5-fold CV)
  ├── Audience Intelligence Agent → Efficiency scoring (CTR × log(Reach))
  ├── Bidding Dynamics Agent      → Win-rate/competition analysis
  └── Fatigue Detection Agent     → Exp decay + CUSUM change-point

Stage 2: Brief Agent → synthesises specialist outputs

Stage 3: Critic Agent → Isolation Forest + adversarial LLM review

Stage 4: Ensemble Engine → confidence-weighted soft voting

Stage 5: Orchestrator → final executive recommendation
```

## License

MIT
