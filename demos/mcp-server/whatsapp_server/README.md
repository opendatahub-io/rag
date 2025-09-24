# WhatsApp RedBank Bot

A FastAPI server that integrates with Evolution API to listen for WhatsApp messages from the RedBank group and automatically responds with "Welcome to RedBank".

## Features

- 🚀 FastAPI-based webhook server
- 📱 Evolution API integration for WhatsApp
- 🎯 Targeted group message filtering (RedBank group)
- 💬 Automatic welcome message responses
- 🔧 Configurable via environment variables
- 📊 Health check and monitoring endpoints
- 🐛 Comprehensive logging

## Setup

### 1. Install Dependencies

```bash
pip install -r whatsapp_requirements.txt
```

### 2. Configure Environment

Copy the example environment file and configure your Evolution API credentials:

```bash
cp whatsapp_server/env.example whatsapp_server/.env
```

Edit `whatsapp_server/.env` with your actual Evolution API credentials:

```env
# Evolution API Configuration
EVOLUTION_API_URL=https://api.evoapicloud.com
EVOLUTION_API_ID=your-instance-id
EVOLUTION_API_TOKEN=your-api-token

# Server Configuration
HOST=0.0.0.0
PORT=8000

# WhatsApp Configuration
REDBANK_GROUP_NAME=RedBank
WELCOME_MESSAGE=Welcome to RedBank
```

### 3. Evolution API Setup

Before running the server, you need to:

1. **Create an Evolution API instance** using your credentials
2. **Connect your WhatsApp** to the instance (scan QR code)
3. **Set up a webhook** to point to your server

#### Set Webhook (Example using curl):

```bash
curl -X POST "https://api.evoapicloud.com/webhook/set/YOUR_INSTANCE_ID" \
  -H "apikey: YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "webhook": {
      "enabled": true,
      "url": "http://your-server-domain:8000/webhook",
      "byEvents": false,
      "base64": false,
      "events": [
        "MESSAGES_UPSERT",
        "MESSAGES_SET"
      ]
    }
  }'
```

## Usage

### Start the Server

```bash
python run_whatsapp_server.py
```

The server will start on `http://0.0.0.0:8000` by default.

### API Endpoints

- **GET `/`** - Health check and basic info
- **GET `/health`** - Detailed health check with Evolution API status
- **POST `/webhook`** - Webhook endpoint for Evolution API (configured automatically)
- **GET `/groups`** - List all groups (debugging)
- **POST `/test-message`** - Send test message

### Testing

#### Test Message Endpoint:

```bash
curl -X POST "http://localhost:8000/test-message" \
  -H "Content-Type: application/json" \
  -d '{
    "number": "5511999999999@s.whatsapp.net",
    "text": "Test message from RedBank bot"
  }'
```

#### Check Groups:

```bash
curl "http://localhost:8000/groups"
```

## How It Works

1. **Webhook Reception**: Evolution API sends webhook notifications when messages are received
2. **Message Filtering**: The server filters for messages from the RedBank group specifically
3. **Auto-Response**: When a message is received from RedBank group, it automatically replies with "Welcome to RedBank"
4. **Background Processing**: Message handling runs in background tasks to avoid blocking the webhook response

## Message Flow

```
WhatsApp Message → Evolution API → Webhook → FastAPI Server → Filter RedBank Group → Send Welcome Reply
```

## Configuration Options

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `EVOLUTION_API_URL` | Evolution API base URL | Required |
| `EVOLUTION_API_ID` | Your Evolution API instance ID | Required |
| `EVOLUTION_API_TOKEN` | Your Evolution API token | Required |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `REDBANK_GROUP_NAME` | Target group name to monitor | `RedBank` |
| `WELCOME_MESSAGE` | Auto-reply message | `Welcome to RedBank` |

## Logging

The server provides comprehensive logging:
- Info level: Normal operations, message processing
- Error level: API errors, webhook processing errors
- Debug level: Detailed message content, ignored events

## Troubleshooting

### Common Issues

1. **Webhook not receiving messages**:
   - Check that webhook URL is accessible from Evolution API servers
   - Verify webhook is properly configured in Evolution API
   - Check server logs for incoming requests

2. **RedBank group not found**:
   - Use `/groups` endpoint to list all available groups
   - Verify group name matches `REDBANK_GROUP_NAME` setting
   - Check that WhatsApp instance is properly connected

3. **Messages not being sent**:
   - Verify Evolution API credentials are correct
   - Check that WhatsApp instance is connected and online
   - Review server logs for API errors

### Debug Mode

Run with debug logging:

```bash
export LOG_LEVEL=DEBUG
python run_whatsapp_server.py
```

## Security Notes

- Keep your Evolution API credentials secure
- Use HTTPS in production
- Consider rate limiting for webhook endpoints
- Monitor server logs for suspicious activity

## Production Deployment

For production deployment:

1. Use a process manager (PM2, systemd, etc.)
2. Set up reverse proxy (nginx)
3. Use HTTPS/SSL certificates
4. Configure proper firewall rules
5. Set up monitoring and alerting
6. Use environment-specific configuration files
