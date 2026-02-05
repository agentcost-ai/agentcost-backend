# AgentCost Backend

FastAPI backend for the AgentCost LLM cost tracking platform.

## Quick Start

```bash
cd agentcost-backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

## Project Structure

```
agentcost-backend/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Environment configuration
│   ├── database.py          # Async SQLAlchemy setup
│   ├── models/
│   │   ├── db_models.py     # SQLAlchemy models
│   │   └── schemas.py       # Pydantic schemas
│   ├── routes/
│   │   ├── events.py        # Event ingestion
│   │   ├── analytics.py     # Analytics queries
│   │   ├── projects.py      # Project management
│   │   └── optimizations.py # Cost optimization suggestions
│   ├── services/
│   │   ├── event_service.py      # Event business logic
│   │   ├── analytics_service.py  # Analytics queries
│   │   └── optimization_service.py # Optimization engine
│   └── utils/
│       └── auth.py          # API key validation
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## API Endpoints

### Health

| Method | Endpoint     | Description  |
| ------ | ------------ | ------------ |
| GET    | `/v1/health` | Health check |

### Projects

| Method | Endpoint            | Description        |
| ------ | ------------------- | ------------------ |
| POST   | `/v1/projects`      | Create new project |
| GET    | `/v1/projects/{id}` | Get project info   |

### Events

| Method | Endpoint           | Description            |
| ------ | ------------------ | ---------------------- |
| POST   | `/v1/events/batch` | Ingest batch of events |

### Analytics

| Method | Endpoint                   | Description                          |
| ------ | -------------------------- | ------------------------------------ |
| GET    | `/v1/analytics/overview`   | Cost overview (total, calls, tokens) |
| GET    | `/v1/analytics/agents`     | Per-agent breakdown                  |
| GET    | `/v1/analytics/models`     | Per-model breakdown                  |

### Pricing (Dynamic - No Auth Required)

| Method | Endpoint                    | Description                      |
| ------ | --------------------------- | -------------------------------- |
| GET    | `/v1/pricing`               | Get all model pricing            |
| GET    | `/v1/pricing/{model}`       | Get pricing for specific model   |
| POST   | `/v1/pricing`               | Update pricing (admin)           |
| POST   | `/v1/pricing/sync-defaults` | Sync default pricing to database |
| GET    | `/v1/analytics/timeseries` | Time series data                     |
| GET    | `/v1/analytics/full`       | Complete analytics response          |

### Optimizations

| Method | Endpoint                    | Description                       |
| ------ | --------------------------- | --------------------------------- |
| GET    | `/v1/optimizations`         | Get cost optimization suggestions |
| GET    | `/v1/optimizations/summary` | Summary of potential savings      |

## Configuration

Create `.env` file:

```bash
# Environment (development, staging, production)
ENVIRONMENT=development

# Database (SQLite for dev, PostgreSQL for prod)
DATABASE_URL=sqlite+aiosqlite:///./agentcost.db
# DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/agentcost

# Authentication (REQUIRED in production!)
# Generate: python -c "import secrets; print(secrets.token_urlsafe(32))"
SECRET_KEY=your-secret-key

# CORS origins (add your frontend domain)
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]

# Debug mode
DEBUG=false
```

## Database

### Development (SQLite)

Works out of the box - database created automatically.

### Production (PostgreSQL)

```sql
CREATE DATABASE agentcost;
CREATE USER agentcost_user WITH ENCRYPTED PASSWORD 'secure-password';
GRANT ALL PRIVILEGES ON DATABASE agentcost TO agentcost_user;
```

```bash
DATABASE_URL=postgresql+asyncpg://agentcost_user:secure-password@localhost:5432/agentcost
```

## Docker

```bash
# Build
docker build -t agentcost-backend .

# Run
docker run -d -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e SECRET_KEY=your-secret-key \
  -e DATABASE_URL=postgresql+asyncpg://... \
  agentcost-backend
```

Or with Docker Compose:

```bash
docker-compose up -d
```

## Authentication

All `/v1/analytics/*` and `/v1/events/*` endpoints require API key:

```bash
curl -H "Authorization: Bearer sk_your_project_api_key" \
  http://localhost:8000/v1/analytics/overview
```

## Example Responses

### Analytics Overview

```json
{
  "total_cost": 45.32,
  "total_calls": 2150,
  "total_tokens": 1250000,
  "avg_cost_per_call": 0.021,
  "avg_latency_ms": 850.5,
  "success_rate": 99.5,
  "period_start": "2026-01-16T00:00:00Z",
  "period_end": "2026-01-23T00:00:00Z"
}
```

### Optimization Suggestion

```json
{
  "type": "model_downgrade",
  "title": "Switch router-agent from gpt-4 to gpt-3.5-turbo",
  "description": "Agent 'router-agent' uses gpt-4 but generates only 50 tokens on average.",
  "estimated_savings_monthly": 45.5,
  "estimated_savings_percent": 95.0,
  "priority": "high",
  "action_items": [
    "Review prompts and outputs",
    "Test with gpt-3.5-turbo",
    "Update model configuration"
  ]
}
```

## Testing

```bash
# Run with test database
DATABASE_URL=sqlite+aiosqlite:///./test.db pytest tests/ -v
```

## Production Checklist

- [ ] Set `ENVIRONMENT=production`
- [ ] Set `SECRET_KEY` (required!)
- [ ] Use PostgreSQL
- [ ] Configure CORS for your domain
- [ ] Set up HTTPS (nginx/Caddy)
- [ ] Enable monitoring (Prometheus)
- [ ] Set up log aggregation


## License

MIT License
