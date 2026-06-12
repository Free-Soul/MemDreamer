#!/bin/bash
# Stop all processes launched by retrieve pipeline
# Kills: retrieve.run agents, embedding servers

echo "Stopping retrieve agent processes..."
pkill -f "retrieve.run" && echo "  Done." || echo "  No processes found."

echo "Stopping embedding_server processes..."
pkill -f "retrieve/scripts/embedding_server" && echo "  Done." || echo "  No processes found."

# Verify
REMAINING=$(ps aux | grep -E "retrieve\.run|embedding_server" | grep -v grep | wc -l)
echo "Remaining processes: $REMAINING"
