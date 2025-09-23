# RedBank Financials MCP Server - Docker Deployment

This Docker setup provides a complete deployment of the RedBank Financials MCP Server with PostgreSQL database.

## Quick Start

1. **Deploy the services:**
   ```bash
   ./deploy.sh
   ```

2. **Or manually with docker-compose:**
   ```bash
   docker-compose up --build -d
   ```

## Services

- **PostgreSQL Database**: `localhost:5432`
  - Database: `bank_statements`
  - User: `postgres`
  - Password: `postgres`

- **MCP Server**: `http://localhost:8000/mcp`
  - RedBank Financials MCP API endpoint

## Available MCP Tools

1. **get_user_by_phone(phone_number: str)**
   - Get user details by phone number
   - Example: `"+1-555-123-4567"`

2. **get_statements(user_id: int)**
   - Get bank statements for a specific user
   - Returns list of statements with totals

3. **get_transactions(statement_id: int)**
   - Get transactions for a specific statement
   - Returns detailed transaction history

## Sample Data

The database is pre-populated with:
- 4 sample users with phone numbers
- 6 bank statements across different users
- 30+ realistic transactions (salary, bills, purchases, etc.)

## Management Commands

```bash
# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Restart services
docker-compose restart

# Access PostgreSQL directly
docker exec -it redbank_postgres psql -U postgres -d bank_statements
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐
│   MCP Server    │───▶│   PostgreSQL     │
│   (Port 8000)   │    │   (Port 5432)    │
│                 │    │                  │
│ - get_user_by_  │    │ - users          │
│   phone         │    │ - statements     │
│ - get_statements│    │ - transactions   │
│ - get_transactions   │                  │
└─────────────────┘    └──────────────────┘
```

## Troubleshooting

- **Database connection issues**: Ensure PostgreSQL is healthy with `docker-compose ps`
- **MCP server not starting**: Check logs with `docker-compose logs mcp-server`
- **Port conflicts**: Modify ports in `docker-compose.yml` if 5432 or 8000 are in use
