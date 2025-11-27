# Build stage
FROM golang:1.22-alpine AS builder

WORKDIR /app

# Install dependencies
RUN apk add --no-cache git

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download all dependencies
RUN go mod download

# Copy the source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -o mcp-jolt-server ./cmd/server

# Final stage
FROM alpine:latest

WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/mcp-jolt-server .

# Copy example files (optional, can be mounted at runtime)
# COPY input.json jolt_spec.json ./  # Optional example files; can be mounted at runtime

# Expose the port the app runs on
EXPOSE 8081

# Command to run the binary
CMD ["./mcp-jolt-server", "server"]
