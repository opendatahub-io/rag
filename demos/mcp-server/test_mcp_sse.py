#!/usr/bin/env python3

"""
Test client for RedBank MCP Server using SSE (Server-Sent Events)
"""

import asyncio
import json
import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client

async def test_mcp_sse():
    """Test the MCP server using SSE transport"""
    
    server_url = "http://localhost:8000/sse"
    
    print("🏦 Testing RedBank MCP Server with SSE")
    print(f"📡 Connecting to: {server_url}")
    print("=" * 50)
    
    try:
        async with sse_client(server_url) as (read, write):
            async with ClientSession(read, write) as session:
                
                # Initialize the session
                print("🔧 Initializing MCP session...")
                await session.initialize()
                print("✅ MCP session initialized successfully")
                
                # List available tools
                print("\n📋 Listing available tools...")
                tools_result = await session.list_tools()
                print(f"✅ Found {len(tools_result.tools)} tools:")
                
                for tool in tools_result.tools:
                    print(f"🔧 {tool.name}")
                    print(f"   Description: {tool.description}")
                    if tool.inputSchema and 'properties' in tool.inputSchema:
                        print("   Parameters:")
                        for param_name, param_info in tool.inputSchema['properties'].items():
                            param_type = param_info.get('type', 'unknown')
                            param_desc = param_info.get('description', 'No description')
                            print(f"     - {param_name} ({param_type}): {param_desc}")
                    print()
                
                # Test get_user_by_phone tool
                print("🧪 Testing get_user_by_phone tool...")
                user_result = await session.call_tool(
                    "get_user_by_phone",
                    {"phone_number": "+1-555-123-4567"}
                )
                
                if user_result.content:
                    user_data = json.loads(user_result.content[0].text)
                    print("✅ User found:")
                    print(f"   Name: {user_data.get('name', 'N/A')}")
                    print(f"   Phone: {user_data.get('phone_number', 'N/A')}")
                    print(f"   Address: {user_data.get('address', 'N/A')}")
                    
                    # Test get_statements tool
                    user_id = user_data.get('user_id')
                    if user_id:
                        print(f"\n🧪 Testing get_statements for user ID {user_id}...")
                        statements_result = await session.call_tool(
                            "get_statements",
                            {"user_id": user_id}
                        )
                        
                        if statements_result.content:
                            statements_data = json.loads(statements_result.content[0].text)
                            if isinstance(statements_data, dict) and 'result' in statements_data:
                                statements_list = statements_data['result']
                            else:
                                statements_list = statements_data
                            
                            print(f"✅ Found {len(statements_list)} statements:")
                            for stmt in statements_list[:2]:  # Show first 2
                                print(f"   Statement {stmt.get('statement_id')}: ${stmt.get('total')} on {stmt.get('date')}")
                            
                            # Test get_transactions tool
                            if statements_list:
                                stmt_id = statements_list[0].get('statement_id')
                                print(f"\n🧪 Testing get_transactions for statement ID {stmt_id}...")
                                transactions_result = await session.call_tool(
                                    "get_transactions",
                                    {"statement_id": stmt_id}
                                )
                                
                                if transactions_result.content:
                                    transactions_data = json.loads(transactions_result.content[0].text)
                                    if isinstance(transactions_data, dict) and 'result' in transactions_data:
                                        transactions_list = transactions_data['result']
                                    else:
                                        transactions_list = transactions_data
                                    
                                    print(f"✅ Found {len(transactions_list)} transactions:")
                                    for txn in transactions_list[:3]:  # Show first 3
                                        print(f"   {txn.get('description')}: ${txn.get('price')} on {txn.get('date')}")
                
                print("\n🎉 All SSE tests completed successfully!")
                    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False
    
    return True


if __name__ == "__main__":
    print("Starting MCP SSE client test...")
    success = asyncio.run(test_mcp_sse())
    if success:
        print("\n✅ MCP SSE test completed successfully!")
    else:
        print("\n❌ MCP SSE test failed!")
