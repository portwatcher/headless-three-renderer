# syntax=docker/dockerfile:1

ARG NODE_VERSION=24-bookworm-slim

FROM node:${NODE_VERSION} AS base

ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0 \
    PNPM_HOME=/pnpm \
    PATH=/pnpm:$PATH

WORKDIR /app

RUN corepack enable && corepack prepare pnpm@9.15.0 --activate

FROM base AS builder

ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        pkg-config \
        python3 \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY packages/renderer/package.json ./packages/renderer/
RUN pnpm install --frozen-lockfile

COPY . .
RUN pnpm --filter @headless-three/renderer build \
    && rm -rf packages/renderer/target target /usr/local/cargo/git /usr/local/cargo/registry

FROM base AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libegl1 \
        libgl1 \
        libvulkan1 \
        mesa-vulkan-drivers \
        vulkan-tools \
    && mkdir -p /tmp/runtime-root \
    && chmod 700 /tmp/runtime-root \
    && rm -rf /var/lib/apt/lists/*

ENV LIBGL_ALWAYS_SOFTWARE=1 \
    WGPU_BACKEND=vulkan \
    XDG_RUNTIME_DIR=/tmp/runtime-root

COPY --from=builder /app /app

CMD ["pnpm", "--filter", "@headless-three/renderer", "test"]
