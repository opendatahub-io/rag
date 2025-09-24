# Simple RedBank WhatsApp Bot 🤖

A simple Python script that polls the Evolution API for new messages in the RedBank WhatsApp group and automatically replies with "Welcome to RedBank".

## Features

- 🔍 **Auto-discovery**: Automatically finds the RedBank group
- 📨 **Message Polling**: Checks for new messages every 10 seconds
- 💬 **Auto-Reply**: Responds with "Welcome to RedBank" to any new message
- 🚫 **Duplicate Prevention**: Tracks processed messages to avoid spam
- 📝 **Comprehensive Logging**: Logs to both console and file
- ⚙️ **Configurable**: Easy configuration via environment variables

## Quick Setup

### 1. Run Setup Script
```bash
./setup_simple_bot.sh
```

### 2. Configure Credentials
Edit `whatsapp_server/.env` with your Evolution API credentials:
```env
EVOLUTION_API_URL=https://api.evoapicloud.com
EVOLUTION_API_ID=your-instance-id-here
EVOLUTION_API_TOKEN=your-token-here
```

### 3. Run the Bot
```bash
source bot_venv/bin/activate
python redbank_bot.py
```

## How It Works

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   RedBank Bot   │───▶│ Evolution API │───▶│ WhatsApp Group  │
│  (Every 10s)    │    │   (Polling)   │    │   (RedBank)     │
└─────────────────┘    └──────────────┘    └─────────────────┘
         ▲                                           │
         │                Welcome to RedBank        │
         └───────────────────────────────────────────┘
```

## Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `EVOLUTION_API_URL` | Evolution API base URL | `https://api.evoapicloud.com` |
| `EVOLUTION_API_ID` | Your instance ID | Required |
| `EVOLUTION_API_TOKEN` | Your API token | Required |
| `REDBANK_GROUP_NAME` | Group name to monitor | `RedBank` |
| `WELCOME_MESSAGE` | Auto-reply message | `Welcome to RedBank` |
| `POLL_INTERVAL` | Polling interval in seconds | `10` |

## What the Bot Does

1. **🔍 Group Discovery**: Finds the RedBank group automatically by name
2. **📨 Message Polling**: Every 10 seconds, fetches recent messages from the group
3. **🧹 Message Filtering**: Only processes new messages (not from bot itself)
4. **💬 Auto-Reply**: Sends "Welcome to RedBank" for each new message
5. **📊 Logging**: Logs all activity with timestamps and details

## Sample Output

```
2024-01-15 10:30:00 - INFO - 🤖 RedBank Bot initialized
2024-01-15 10:30:00 - INFO - 📱 Instance ID: e940c744-7903-478b-b3fa-6c14b68f74d0
2024-01-15 10:30:00 - INFO - 🎯 Target Group: redbank
2024-01-15 10:30:00 - INFO - 💬 Welcome Message: Welcome to RedBank
2024-01-15 10:30:01 - INFO - ✅ Found RedBank group: RedBank Official (120363025@g.us)
2024-01-15 10:30:01 - INFO - 🔄 Starting polling loop (every 10s)
2024-01-15 10:30:15 - INFO - 📨 New message from John: Hello everyone!
2024-01-15 10:30:16 - INFO - ✅ Sent welcome message to RedBank group
2024-01-15 10:30:16 - INFO - 📊 Processed 1 new messages
```

## Requirements

- Python 3.7+
- Evolution API account with active WhatsApp instance
- RedBank group must exist and bot instance must be a member

## Files Created

- `redbank_bot.log` - Detailed bot activity log
- `bot_venv/` - Python virtual environment (if using setup script)

## Troubleshooting

### Bot can't find RedBank group
- Check that your WhatsApp instance is connected
- Verify the group name matches `REDBANK_GROUP_NAME` setting
- Make sure your bot instance is a member of the group

### No messages being processed
- Check that `EVOLUTION_API_ID` and `EVOLUTION_API_TOKEN` are correct
- Verify your Evolution API instance is active
- Check the log file for error details

### Bot responding to old messages
- The bot filters messages from the last 5 minutes on startup
- Restart the bot to reset the message tracking

## Stopping the Bot

Press `Ctrl+C` to stop the bot gracefully:
```
^C2024-01-15 10:35:00 - INFO - 👋 Bot stopped by user
```

## Advanced Usage

### Custom Poll Interval
```bash
export POLL_INTERVAL=5  # Poll every 5 seconds
python redbank_bot.py
```

### Custom Welcome Message
```bash
export WELCOME_MESSAGE="Hello! Welcome to our RedBank community 🏦"
python redbank_bot.py
```

### Different Group Name
```bash
export REDBANK_GROUP_NAME="RedBank Official"
python redbank_bot.py
```

This simple bot is perfect for basic auto-reply functionality without the complexity of webhook servers! 🎉
