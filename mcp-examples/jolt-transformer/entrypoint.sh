#!/bin/sh
set -e

# If the first argument is a known subcommand, run the binary with that subcommand
if [ "$1" = "mcp" ] || [ "$1" = "mcp-sse" ] || [ "$1" = "server" ] || [ "$1" = "transform" ]; then
    exec ./mcp-jolt-server "$@"
fi

# Default to running the server if no known subcommand is provided
# This allows passing flags directly to the server command
exec ./mcp-jolt-server server "$@"
