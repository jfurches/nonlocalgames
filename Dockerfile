FROM ghcr.io/astral-sh/uv:debian-slim

# Create a new user to avoid root access
RUN useradd -ms /bin/sh user
USER user

# Copy project files
WORKDIR /app
COPY --chown=user adapt-gym /app/adapt-gym
COPY --chown=user nonlocalgames /app/nonlocalgames

# Install all nonlocalgame dependencies, minus pytest and stuff
WORKDIR /app/nonlocalgames
RUN uv sync --frozen --extra adapt --no-dev
RUN uv pip install jupyterlab notebook

# Add extra plotting utils
RUN uv pip install matplotlib seaborn pandas

# Run jupyter at /notebooks directory
EXPOSE 8888
ENTRYPOINT ["uv", "run", "jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--notebook-dir=/notebooks"]