## Getting started with MCP

> this tutorial runs on MAC

1. Clone the repo:
```git clone https://github.com/Schimuneck/llamastack-rag-mcp-demo```

2. Create local venv and activate it:
```cd llamastack-rag-mcp-demo```

```uv venv .venv --python 3.12```

```source .venv/bin/activate```

3. Install requirements:
```uv pip install -r requirements.txt```

4. Start your local Postgres server:
```brew services start postgresql```

5. Allow permission to execute scripts to create PostgreSQL database and insert data:
```chmod +x scripts/create_db_and_insert_data.sh```

6. Run script to create PostgreSQL db and insert data
```./scripts/create_db_and_insert_data.sh```

7. Get data
```psql -d electroshop_sales -c "SELECT * FROM customers;"```

8. Add MCP server to Cursor settings and enable MCP tools:

Eg. json file:
```
{
  "mcpServers": {
    "mcp-for-electroshop-sales-server": {
      "command": "/Users/iamiller/GitHub/llamastack-rag-mcp-demo/.venv/bin/python",
      "args": [
        "/Users/iamiller/GitHub/llamastack-rag-mcp-demo/mcp_server/mcp_server.py"
      ],
      "description": "MCP server for ElectroShop Sales"
    }
  }
}
```

9. Run MCP server:

```cd mcp_server```

```uv run python mcp_server.py```
