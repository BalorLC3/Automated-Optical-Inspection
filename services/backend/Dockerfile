# Build Stage
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod ./
# COPY go.sum ./ # TODO: Uncomment this after run 'go mod tidy' once
COPY main.go .
RUN go build -o main .

# Run Stage
FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/main .

# Create data directories
RUN mkdir -p ./data/uploads

EXPOSE 8080
CMD ["./main"]