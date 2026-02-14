# AgentCost Backend

FastAPI backend for the AgentCost LLM cost tracking and analytics platform.

## Quick Start

```bash
cd agentcost-backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Interactive docs at `https://agentcost.tech/docs/sdk`

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
│   │   ├── auth.py          # User authentication and registration
│   │   ├── members.py       # Project member management
│   │   ├── events.py        # Event ingestion
│   │   ├── analytics.py     # Analytics queries
│   │   ├── projects.py      # Project management
│   │   ├── optimizations.py # Cost optimization suggestions
│   │   ├── pricing.py       # Dynamic model pricing
│   │   ├── feedback.py      # User feedback system
│   │   ├── attachments.py   # File upload/download
│   │   └── admin/           # Admin platform management
│   ├── services/
│   │   ├── auth_service.py      # Authentication business logic
│   │   ├── member_service.py    # Project membership logic
│   │   ├── event_service.py     # Event processing
│   │   ├── analytics_service.py # Analytics queries
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

### Authentication

| Method | Endpoint                    | Description                      |
| ------ | --------------------------- | -------------------------------- |
| POST   | `/v1/auth/register`         | User registration                |
| POST   | `/v1/auth/login`            | User login                       |
| POST   | `/v1/auth/logout`           | User logout                      |
| POST   | `/v1/auth/refresh`          | Refresh access token             |
| POST   | `/v1/auth/forgot-password`  | Request password reset           |
| POST   | `/v1/auth/reset-password`   | Confirm password reset           |
| GET    | `/v1/auth/me`               | Get current user profile         |
| PUT    | `/v1/auth/me`               | Update user profile              |
| POST   | `/v1/auth/verify-email`     | Verify email address             |
| POST   | `/v1/auth/resend-verification` | Resend verification email     |

### Projects

| Method | Endpoint                    | Description                      |
| ------ | --------------------------- | -------------------------------- |
| POST   | `/v1/projects`              | Create new project               |
| GET    | `/v1/projects`              | List user projects               |
| GET    | `/v1/projects/{id}`         | Get project details              |
| PUT    | `/v1/projects/{id}`         | Update project                   |
| DELETE | `/v1/projects/{id}`         | Delete project                   |
| POST   | `/v1/projects/{id}/invite`  | Invite member to project         |
| GET    | `/v1/projects/{id}/members` | List project members             |
| PUT    | `/v1/projects/{id}/members/{user_id}` | Update member role        |
| DELETE | `/v1/projects/{id}/members/{user_id}` | Remove project member     |

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
| GET    | `/v1/analytics/timeseries` | Time series data                     |
| GET    | `/v1/analytics/full`       | Complete analytics response          |

### Pricing (Dynamic - No Auth Required)

| Method | Endpoint                    | Description                      |
| ------ | --------------------------- | -------------------------------- |
| GET    | `/v1/pricing`               | Get all model pricing            |
| GET    | `/v1/pricing/{model}`       | Get pricing for specific model   |
| POST   | `/v1/pricing`               | Update pricing (admin)           |
| POST   | `/v1/pricing/sync-defaults` | Sync default pricing to database |

### Optimizations

| Method | Endpoint                    | Description                       |
| ------ | --------------------------- | --------------------------------- |
| GET    | `/v1/optimizations`         | Get cost optimization suggestions |
| GET    | `/v1/optimizations/summary` | Summary of potential savings      |

### Feedback

| Method | Endpoint                    | Description                       |
| ------ | --------------------------- | --------------------------------- |
| POST   | `/v1/feedback`              | Submit feedback                   |
| GET    | `/v1/feedback`              | List feedback (admin)             |
| GET    | `/v1/feedback/{id}`         | Get feedback details              |
| PUT    | `/v1/feedback/{id}`         | Update feedback (admin)           |
| POST   | `/v1/feedback/{id}/upvote`  | Upvote feedback                   |
| POST   | `/v1/feedback/{id}/comment` | Add comment to feedback           |

### Attachments

| Method | Endpoint                    | Description                       |
| ------ | --------------------------- | --------------------------------- |
| POST   | `/v1/attachments`           | Upload file                       |
| GET    | `/v1/attachments/{id}`      | Download file                     |
| DELETE | `/v1/attachments/{id}`      | Delete file                       |

### Admin (Superuser Only)

| Method | Endpoint                    | Description                       |
| ------ | --------------------------- | --------------------------------- |
| GET    | `/v1/admin/overview`        | Platform-wide statistics          |
| GET    | `/v1/admin/users`           | User management                   |
| GET    | `/v1/admin/projects`        | Project governance                |
| GET    | `/v1/admin/analytics`       | Cross-tenant analytics            |
| GET    | `/v1/admin/feedback`        | Feedback triage                   |
| GET    | `/v1/admin/audit-log`       | Admin action audit trail          |

## Configuration

Create `.env` file:

```bash
# Environment (development, staging, production)
ENVIRONMENT=development

# Database (SQLite for dev, PostgreSQL for prod)
DATABASE_URL=sqlite+aiosqlite:///./agentcost.db
# DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/agentcost

# Authentication
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key-here
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# Email (required for user registration)
RESEND_API_KEY=your-resend-api-key
FROM_EMAIL=noreply@yourdomain.com

# CORS origins (add your frontend domains)
CORS_ORIGINS=["http://localhost:3000","http://localhost:3001","https://yourdomain.com"]

# File uploads
UPLOAD_DIR=
MAX_UPLOAD_SIZE_MB=10

# Admin auto-seed (for first-time setup)
ADMIN_EMAIL=
ADMIN_PASSWORD=

# Debug mode
DEBUG=false

# Auto-sync pricing on startup
AUTO_SYNC_PRICING_ON_STARTUP=true
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

The API supports two authentication methods:

### 1. User Authentication (JWT)

For user-facing operations like project management and feedback:

```bash
# Register/Login to get JWT token
curl -X POST http://localhost:8000/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password"}'

# Use JWT token for authenticated requests
curl -H "Authorization: Bearer your_jwt_token" \
  http://localhost:8000/v1/projects
```

### 2. API Key Authentication

For SDK event ingestion and analytics:

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
- [ ] Set `SECRET_KEY` and `JWT_SECRET_KEY` (required!)
- [ ] Configure email service (Resend API key)
- [ ] Use PostgreSQL database
- [ ] Configure CORS for your domains
- [ ] Set up file storage (local or cloud)
- [ ] Enable HTTPS (nginx/Caddy)
- [ ] Set up monitoring (Prometheus)
- [ ] Configure log aggregation
- [ ] Set up backup strategy for database and uploads


## License

MIT License
