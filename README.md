# Indian Stock Analyst Agent

An AI-powered **advanced multi-agent** stock analysis system for Indian markets (NSE & BSE) built with OpenAI Agents SDK. This is the most comprehensive stock analysis system available, featuring 10 specialized agents coordinated by an Orchestrator to provide thorough BUY/SELL/HOLD recommendations with professional PDF reports.

## Command to start
venv\Scripts\python.exe main.py

## Architecture

```
                                    ┌─────────────────────────────────────┐
                                    │       ORCHESTRATOR AGENT            │
                                    │   Coordinates all analysis phases   │
                                    └─────────────────┬───────────────────┘
                                                      │
                    ┌─────────────────────────────────┼─────────────────────────────────┐
                    │                                 │                                 │
          ┌─────────▼─────────┐              ┌────────▼────────┐             ┌──────────▼──────────┐
          │   ANALYSIS TEAM   │              │   DEBATE TEAM   │             │    RISK TEAM        │
          └─────────┬─────────┘              └────────┬────────┘             └──────────┬──────────┘
                    │                                 │                                 │
    ┌───────────────┼───────────────┐         ┌───────┼───────┐               ┌─────────┼─────────┐
    │       │       │       │       │         │       │       │               │                   │
┌───▼───┐┌──▼──┐┌───▼───┐┌──▼──┐┌───▼───┐  ┌──▼──┐┌───▼───┐┌──▼──┐      ┌─────▼─────┐    ┌────────▼────────┐
│ FUND  ││TECH ││ SENT  ││MACRO││  DOC  │  │BULL ││ BEAR  ││JUDGE│      │   RISK    │    │   PORTFOLIO     │
│ANALYST││ANAL ││ANALYST││ANAL ││ANALYST│  │ADVOC││ ADVOC ││     │      │  MANAGER  │    │    ANALYST      │
│       ││     ││       ││     ││       │  │     ││       ││     │      │           │    │                 │
│P/E,ROE││RSI  ││VADER  ││RBI  ││Q.Res  │  │Bull ││Bear   ││Final│      │Position   │    │Health Score     │
│Debt   ││MACD ││TextBlb││FII  ││Peer   │  │Case ││Case   ││Verd.│      │Sizing     │    │Diversification  │
│Growth ││Trend││News   ││DII  ││Announce│ └─────┘└───────┘└─────┘      │Stop Loss  │    │Correlation      │
└───────┘└─────┘└───────┘└─────┘└───────┘                               │Risk-Reward│    │Rebalancing      │
                                                                        └───────────┘    └─────────────────┘
```

## Features

### Core Analysis
- **10-Agent Architecture**: Comprehensive analysis from multiple specialized perspectives
- **Real-time Stock Data**: Current prices, volumes, and market data from NSE/BSE
- **Fundamental Analysis**: P/E, P/B, ROE, debt, growth, peer comparison
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Support/Resistance, Fibonacci
- **Sentiment Analysis**: VADER + TextBlob combined sentiment scoring

### Advanced Features (NEW)
- **Bull/Bear Debate**: Adversarial agents argue both sides for balanced analysis
- **Macro Analysis**: RBI policy, inflation, FII/DII flows, global context
- **Document Analysis**: Quarterly results, company announcements, peer comparison
- **Risk Management**: Position sizing, ATR-based stop loss, risk-reward assessment
- **Portfolio Analysis**: Health score, diversification, correlation, rebalancing
- **Exa Live Research Integration**: Real-time web/company/deep research via MCP HTTP tools

### Output
- **Professional PDF Reports**: Comprehensive reports with all analysis
- **Weighted Scoring System**: Objective recommendations based on multi-factor scoring
- **Position Sizing**: Exact share quantities for your portfolio size

## Project Structure

```
Stock Agent/
├── main.py                 # Entry point - run this to start
├── agent.py                # Multi-agent system configuration
├── config.py               # Configuration settings
├── pdf_generator.py        # PDF report generation
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
│
├── agents/                 # All 10 specialized agents
│   ├── __init__.py
│   ├── orchestrator.py     # Main orchestrator + quick version
│   ├── fundamental_analyst.py  # Fundamental analysis
│   ├── technical_analyst.py    # Technical analysis
│   ├── sentiment_analyst.py    # News sentiment analysis
│   ├── macro_analyst.py        # Macroeconomic analysis (NEW)
│   ├── document_analyst.py     # Document/filing analysis (NEW)
│   ├── bull_agent.py           # Bull advocate (NEW)
│   ├── bear_agent.py           # Bear advocate (NEW)
│   ├── debate_judge.py         # Debate synthesizer (NEW)
│   ├── risk_manager.py         # Risk management (NEW)
│   └── portfolio_analyst.py    # Portfolio analysis (NEW)
│
├── models/                 # Pydantic schemas
│   ├── __init__.py
│   └── schemas.py          # Structured output models
│
├── tools/                  # 30+ analysis tools
│   ├── __init__.py
│   ├── stock_data.py       # Stock data fetching
│   ├── technical_analysis.py # Technical indicators
│   ├── news_fetcher.py     # News fetching
│   ├── sentiment_analyzer.py # VADER + TextBlob sentiment
│   ├── exa_research.py     # Exa MCP HTTP live research (NEW)
│   ├── macro_data.py       # India macro indicators (NEW)
│   ├── portfolio_analyzer.py # Portfolio analysis (NEW)
│   ├── risk_management.py  # Position sizing/stops (NEW)
│   └── document_analyzer.py # Company filings (NEW)
│
├── docs/
│   ├── OPENAI_AGENTS_SDK_REFERENCE.md
│   ├── INDIAN_STOCK_API_REFERENCE.md
│   ├── RESEARCH_ANALYSIS.md
│   └── IMPLEMENTATION_REFERENCE.md
│
└── reports/                # Generated PDF reports
```

## Installation

1. **Clone/Download the project**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your LLM provider**:
   ```bash
   # Option 1: OpenAI
   export LLM_PROVIDER=openai
   export OPENAI_API_KEY=your-api-key-here

   # Option 2: Create .env file and choose another provider
   cp .env.example .env
   # Edit .env, then set LLM_PROVIDER plus the matching provider key.
   ```

### BYOLLM Provider Configuration

The agent supports OpenAI-compatible providers through `LLM_PROVIDER`:

| Provider | Required settings | Example model |
|----------|-------------------|---------------|
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `groq` | `GROQ_API_KEY_PRIMARY` (or `GROQ_API_KEY`) | `llama-3.3-70b-versatile` |
| `openrouter` | `OPENROUTER_API_KEY` | `openai/gpt-4o-mini` |
| `mistral` | `MISTRAL_API_KEY_PRIMARY` (or `MISTRAL_API_KEY`) | `mistral-large-latest` |
| `ollama` | `OLLAMA_BASE_URL` | `llama3.1` |
| `custom` | `LLM_BASE_URL`, `LLM_API_KEY`, `MODEL_NAME` | provider-specific |

Use `python test_api.py --model <model-name>` to verify the selected provider before running a full stock-analysis pipeline. For local Ollama, start Ollama first and keep `OLLAMA_BASE_URL=http://localhost:11434/v1/`.

### Mistral API Key Configuration & Rate-Limit Fallback

The Mistral provider supports up to **three API keys** to work around free-tier rate limits (HTTP 429 errors).

#### Why multiple keys?
Mistral's free tier has strict per-minute request limits. When a key is exhausted, the API returns HTTP 429. The fallback chain lets the agent automatically switch to a fresh key and continue without interruption.

#### How the fallback chain works

```
Request → MISTRAL_API_KEY_PRIMARY
              │
              ├─ Success → done
              └─ HTTP 429 → try MISTRAL_API_KEY_SECONDARY
                                │
                                ├─ Success → done
                                └─ HTTP 429 → try MISTRAL_API_KEY_TERTIARY
                                                  │
                                                  ├─ Success → done
                                                  └─ HTTP 429 → raise error
```

**Key behaviours:**
- Fallback is triggered **only** on HTTP 429 (rate-limit) responses. Any other error (auth failure, network error, etc.) is raised immediately without trying the next key.
- Each key is attempted **at most once** per request — no infinite loops.
- Only the key *slot name* (`primary`, `secondary`, or `tertiary`) is written to logs. The actual key value is **never** logged.

#### Setup

Set the keys in your `.env` file (copy from `.env.example`):

```bash
# Required when LLM_PROVIDER=mistral
MISTRAL_API_KEY_PRIMARY=your_primary_mistral_api_key_here

# Optional — used only if PRIMARY hits a 429 rate-limit
MISTRAL_API_KEY_SECONDARY=your_secondary_mistral_api_key_here

# Optional — used only if SECONDARY also hits a 429 rate-limit
MISTRAL_API_KEY_TERTIARY=your_tertiary_mistral_api_key_here
```

> **Note:** `MISTRAL_API_KEY` (without suffix) is kept for backwards compatibility and acts as an alias for `MISTRAL_API_KEY_PRIMARY` if the primary key is not set separately.

You can obtain free Mistral API keys at <https://console.mistral.ai/>. Create multiple accounts or use the same account's key rotation feature to get additional keys.

### Groq API Key Configuration & Rate-Limit Fallback

The Groq provider supports up to **three API keys** to work around free-tier rate limits (HTTP 429 errors).

#### Why multiple keys?
Groq's free tier has strict per-minute request limits. When a key is exhausted, the API returns HTTP 429. The fallback chain lets the agent automatically switch to a fresh key and continue without interruption.

#### How the fallback chain works

```
Request → GROQ_API_KEY_PRIMARY
              │
              ├─ Success → done
              └─ HTTP 429 → try GROQ_API_KEY_SECONDARY
                                │
                                ├─ Success → done
                                └─ HTTP 429 → try GROQ_API_KEY_TERTIARY
                                                  │
                                                  ├─ Success → done
                                                  └─ HTTP 429 → raise error
```

**Key behaviours:**
- Fallback is triggered **only** on HTTP 429 (rate-limit) responses. Any other error (auth failure, network error, etc.) is raised immediately without trying the next key.
- Each key is attempted **at most once** per request — no infinite loops.
- Only the key *slot name* (`primary`, `secondary`, or `tertiary`) is written to logs. The actual key value is **never** logged.

#### Setup

Set the keys in your `.env` file (copy from `.env.example`):

```bash
# Required when LLM_PROVIDER=groq
GROQ_API_KEY_PRIMARY=your_primary_groq_api_key_here

# Optional — used only if PRIMARY hits a 429 rate-limit
GROQ_API_KEY_SECONDARY=your_secondary_groq_api_key_here

# Optional — used only if SECONDARY also hits a 429 rate-limit
GROQ_API_KEY_TERTIARY=your_tertiary_groq_api_key_here
```

> **Note:** `GROQ_API_KEY` (without suffix) is kept for backwards compatibility and acts as an alias for `GROQ_API_KEY_PRIMARY` if the primary key is not set separately.

You can obtain free Groq API keys at <https://console.groq.com/>. Create multiple accounts to get additional keys for the fallback chain.

## Usage

### Interactive Mode
```bash
python main.py
```
This starts an interactive session where you can ask questions about stocks.

### Single Query Mode
```bash
python main.py "Should I buy RELIANCE?"
python main.py "Analyze TCS for long-term investment"
python main.py "I own INFY at 1500, should I hold or sell?"
```

### Web Chat Mode (HTML Frontend + PDF Links)
```bash
python web_server.py
```
Then open `http://127.0.0.1:8000` in your browser.

Web app features now include:
- User registration/login with role support (`user`, `admin`)
- SQLite persistence (`app_data.db`) for users, sessions, chat history, reports
- Previous chats and generated report history in UI
- Admin panel for account/report/chat statistics

Important defaults:
- If no admin exists, server bootstraps one from env:
  - `ADMIN_USERNAME` (default: `admin`)
  - `ADMIN_PASSWORD` (default: `admin123`, change immediately)

Available web routes:
- `GET /` -> rich chat UI (`chat.html`)
- `POST /api/register` -> create user account
- `POST /api/login` -> login user/admin
- `POST /api/logout` -> logout session
- `GET /api/me` -> current session user
- `POST /api/chat` -> run analysis + generate PDF + persist history
- `POST /api/chat/start` -> start async analysis job and return `job_id`
- `GET /api/chat/status?job_id=...` -> poll live progress/status/result
- `GET /api/chats` -> previous chat history for logged-in user
- `GET /api/reports` -> previous generated reports (all for admin, own for user)
- `GET /api/prompt-examples` -> suggested prompts
- `GET /api/admin/stats` -> admin stats
- `GET /api/admin/users` -> admin user list
- `GET /reports/<filename>.pdf` -> open generated report (auth + ownership enforced)
- `GET /health` -> basic server health (now includes Exa MCP config visibility)

### Example Queries

**Stock Analysis:**
- "Should I buy RELIANCE?"
- "Analyze TCS stock fundamentals and technicals"
- "Is HDFCBANK a good buy at current levels?"
- "I bought INFY at 1600, current price is 1450. Should I hold or sell?"

**Portfolio Analysis:**
- "Analyze my portfolio: RELIANCE 100 shares at 2500, TCS 50 shares at 3500"
- "Is my portfolio well diversified?"
- "Suggest rebalancing for my holdings"

**Comparisons:**
- "Compare ICICIBANK and AXISBANK"
- "Which is better: TCS or INFY?"

**Specific Analysis:**
- "What are the support and resistance levels for SBIN?"
- "What is the macro environment for IT stocks?"
- "How did TITAN's quarterly results compare to peers?"

## Analysis Workflow

When you analyze a stock, the system runs through 4 phases:

### Phase 1: Data Gathering
1. **Fundamental Analyst** → Valuation, profitability, growth metrics
2. **Technical Analyst** → Price action, indicators, support/resistance
3. **Sentiment Analyst** → News sentiment from multiple sources
3b. **News Intelligence Analyst** → Exa real-time search/snapshot/deep-research + event analysis
4. **Macro Analyst** → RBI policy, inflation, FII/DII flows
5. **Document Analyst** → Quarterly results, peer comparison

### Phase 2: Debate
6. **Bull Advocate** → Builds the strongest case FOR buying
7. **Bear Advocate** → Builds the strongest case AGAINST buying
8. **Debate Judge** → Evaluates both cases, determines winner

### Phase 3: Risk Assessment
9. **Risk Manager** → Position sizing, stop loss, risk-reward ratio

### Phase 4: Synthesis
10. **Orchestrator** → Combines all inputs, generates final recommendation
11. **PDF Report** → Professional report with all analysis

## Scoring System

The final recommendation uses weighted scoring:

| Factor | Weight | Description |
|--------|--------|-------------|
| Fundamental | 30% | P/E, ROE, debt, growth |
| Technical | 25% | RSI, MACD, trend, support |
| Sentiment | 15% | News sentiment score |
| Macro | 15% | Economic environment |
| Debate | 15% | Bull/Bear winner adjustment |

**Recommendation Thresholds:**
- Score >= 8.0: **STRONG BUY**
- Score >= 6.5: **BUY**
- Score >= 4.5: **HOLD**
- Score >= 3.0: **SELL**
- Score < 3.0: **STRONG SELL**

## Supported Stocks

### NSE Stocks
Use the stock symbol directly (e.g., `RELIANCE`, `TCS`, `INFY`)
The agent auto-adds `.NS` suffix for NSE.

### BSE Stocks
Add `.BO` suffix (e.g., `RELIANCE.BO`, `TCS.BO`)

### Popular Stocks
- RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK
- HINDUNILVR, SBIN, BHARTIARTL, ITC, KOTAKBANK
- LT, AXISBANK, ASIANPAINT, MARUTI, TITAN
- SUNPHARMA, BAJFINANCE, WIPRO, HCLTECH

### Indices
- NIFTY50 (^NSEI)
- SENSEX (^BSESN)
- NIFTY BANK (^NSEBANK)

## API Data Sources

- **Stock Data**: Yahoo Finance via yfinance library (free, no API key needed)
- **News**: Google News RSS + Yahoo Finance news
- **Macro Data**: Simulated RBI/government data (extend for real APIs)
- **LLM**: BYOLLM via OpenAI-compatible providers (`openai`, `groq`, `openrouter`, `mistral`, `ollama`, or `custom`)

## ⚠️ Data Limitations & Accuracy

> **IMPORTANT**: This system uses a mix of real-time and simulated data. Understand the differences before making investment decisions.

### Real-Time Data (HIGH Reliability)
These data sources fetch **LIVE data** from APIs:

| Data Type | Source | Freshness |
|-----------|--------|-----------|
| Stock Prices | Yahoo Finance | Real-time (15-min delay) |
| Technical Indicators | Calculated from yfinance | Real-time |
| Nifty/Sensex Benchmark | Yahoo Finance | Real-time |
| Sector Indices | Yahoo Finance | Real-time |
| Global Markets (US, Gold, Oil) | Yahoo Finance | Real-time |
| Stock Fundamentals (P/E, ROE) | Yahoo Finance | Quarterly updated |

### Simulated Data (FOR DEMONSTRATION ONLY)
These data sources use **HARDCODED REFERENCE DATA** and may be outdated:

| Data Type | Last Updated | What to Use Instead |
|-----------|--------------|---------------------|
| India Macro Indicators (RBI rates, inflation) | 2024-12-06 | [RBI Website](https://www.rbi.org.in), [MOSPI](https://www.mospi.gov.in) |
| FII/DII Activity | 2024-12-06 | [NSE FII/DII Reports](https://www.nseindia.com/reports/fii-dii) |
| Company Announcements | 2024-12-06 | [BSE Announcements](https://www.bseindia.com/corporates/ann.html) |
| Management Commentary | 2024-12-06 | Company quarterly earnings calls |
| Peer Comparison Mappings | 2024-12-06 | [Screener.in](https://www.screener.in), [Trendlyne](https://trendlyne.com) |

### How to Identify Simulated Data
All simulated data outputs include a disclaimer block:
```json
{
  "disclaimer": {
    "warning": "SIMULATED DATA - For demonstration only",
    "message": "...",
    "recommendation": "Verify with official sources..."
  }
}
```

### Data Version Tracking
See `config/data_versions.json` for complete tracking of all data sources and their freshness.

### Upgrading to Real Data
To replace simulated data with real APIs:
1. **FII/DII**: Scrape NSE daily reports or use paid data providers
2. **Macro Data**: Integrate RBI API or Trading Economics API
3. **Announcements**: Use BSE/NSE corporate filing APIs

## Configuration

Edit `config.py` to customize:
- Model settings (temperature, max turns)
- Technical analysis parameters
- PDF output directory

## Requirements

- Python 3.10+
- OpenAI API key (with GPT-4o access recommended)
- Internet connection for real-time data

## What Makes This Better Than Others

Based on research of leading platforms (Liquide, TradingAgents, MarketSenseAI):

| Feature | Liquide | TradingAgents | This System |
|---------|---------|---------------|-------------|
| Multi-Agent | No | Yes | Yes (10 agents) |
| Bull/Bear Debate | No | Yes | Yes |
| Portfolio Health | Yes | No | Yes |
| Macro Analysis | No | No | Yes |
| Risk Management | Basic | Yes | Yes (comprehensive) |
| Document Analysis | No | No | Yes |
| Peer Comparison | Basic | No | Yes |
| PDF Reports | No | No | Yes |

## Disclaimer

This tool is for **educational and informational purposes only**. It does not constitute financial advice. Stock investments are subject to market risks. Always consult a qualified financial advisor before making investment decisions.

## License

MIT License
