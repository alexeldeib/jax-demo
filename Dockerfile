FROM ubuntu:22.04

RUN apt update && apt install -yq curl

ENV PATH=/venv/bin:$PATH
ENV PATH=/root/.cargo/bin:$PATH

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# set up a virtual env to use for whatever app is destined for this container.
RUN uv venv --python 3.12.5 /venv

WORKDIR /app

COPY . .
RUN uv sync
