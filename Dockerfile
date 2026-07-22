# syntax=docker/dockerfile:1

# cargo-chef splits the build into a dependency layer and a source layer,
# so an ordinary source edit doesn't force every dependency in this
# workspace (rusqlite's bundled SQLite, ring, tokio-postgres, ...) to
# recompile from scratch -- only `cargo chef cook` below re-runs, and only
# when Cargo.toml/Cargo.lock actually change. Built from the official rust
# image rather than a third-party cargo-chef image, so this only ever
# depends on registries (Docker Hub's library images, crates.io) most
# build environments already allow.
FROM rust:1-bookworm AS chef
RUN cargo install cargo-chef --locked
WORKDIR /app

FROM chef AS planner
COPY Cargo.toml Cargo.lock ./
COPY crates crates
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY Cargo.toml Cargo.lock ./
COPY crates crates
RUN cargo build --release -p rp-server

# debian:bookworm-slim, not distroless -- rustls-native-certs (used for
# both outbound provider TLS and an optional TLS-enabled [persistence]
# Postgres connection) reads the OS trust store at runtime, so
# ca-certificates has to actually be present in this image, not just
# baked into the build stage.
FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --no-create-home --uid 10001 rusty_provider
COPY --from=builder /app/target/release/rp-server /usr/local/bin/rp-server

WORKDIR /app
USER rusty_provider
EXPOSE 8080

# See config.example.toml -- mount your own config.toml (or set
# CONFIG_PATH elsewhere) and provide provider API keys as env vars;
# nothing secret is baked into this image.
ENV CONFIG_PATH=/app/config.toml

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

ENTRYPOINT ["rp-server"]
