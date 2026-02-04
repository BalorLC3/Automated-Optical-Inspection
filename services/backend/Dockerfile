# ---------- Build stage ----------
FROM golang:1.22-alpine AS builder

WORKDIR /app

# Copy module files first (better caching)
COPY go.mod go.sum ./
RUN go mod download

# Copy source
COPY . .

# Build binary
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o main .

# ---------- Runtime stage ----------
FROM alpine:latest

WORKDIR /app

# Copy binary
COPY --from=builder /app/main .

# Create data directories
RUN mkdir -p ./data/uploads

EXPOSE 8080

CMD ["./main"]
