FROM golang:1.22-alpine AS build
WORKDIR /src
COPY . .
RUN go mod init api-gateway || true && go mod tidy && go build -o /out/api-gateway

FROM alpine:3.20
WORKDIR /app
COPY --from=build /out/api-gateway /app/api-gateway
EXPOSE 9000
CMD ["/app/api-gateway"]