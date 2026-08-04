#!/usr/bin/env bash
set -e

if [ ! -f .env ]; then
    cp .env.sample .env
    echo "Created .env from .env.sample."

    read -rsp "Enter your Hugging Face token: " HF_TOKEN
    echo

    sed -i.bak "s/^HF_TOKEN=.*/HF_TOKEN=$HF_TOKEN/" .env
    rm -f .env.bak
fi

docker compose up --build