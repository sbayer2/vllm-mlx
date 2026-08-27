#!/bin/bash
# vllm-mlx Gradio chat UI. Double-clickable, or launched by vllm-mlx-launch.sh.
# Ctrl+C in this window stops the UI; the server keeps running.

SERVED_NAME="${VLLM_MLX_SERVED_NAME:-qwen}"
SERVER_PORT="${VLLM_MLX_PORT:-8000}"
UI_PORT="${VLLM_MLX_UI_PORT:-7860}"
# Thinking and the answer share this budget: with reasoning on, the <think>
# block is spent from the same allowance, so a reply can truncate mid-sentence
# if this is set too tight. 32768 matches the server's own generation cap, so
# this is the ceiling rather than a compromise.
MAX_TOKENS="${VLLM_MLX_MAX_TOKENS:-32768}"
# Must be at least the server's --timeout, which reads the same variable.
# A shorter client deadline reports "server took too long" for a request the
# server is still working on, and throws the work away.
REQUEST_TIMEOUT="${VLLM_MLX_TIMEOUT:-1800}"
VENV="${VLLM_MLX_VENV:-$HOME/vllm-mlx-env}"

source "$VENV/bin/activate" || {
    echo "Could not activate $VENV — is vllm-mlx installed there?"
    read -r -p "Press return to close. "
    exit 1
}

echo "Chat UI on port $UI_PORT -> server http://127.0.0.1:$SERVER_PORT (model '$SERVED_NAME')"
echo "Type a question alongside any attachment — an empty textbox sends no text"
echo "part, and the model answers in Chinese for lack of a language cue."
echo

# Job control puts the UI in its own process group so the trap below decides
# what it receives, rather than the terminal signalling it directly.
set -m

# --served-model-name must match the server's, or every reply is
# "Error: The model 'default' does not exist."
vllm-mlx-chat \
    --server-url "http://127.0.0.1:$SERVER_PORT" \
    --served-model-name "$SERVED_NAME" \
    --max-tokens "$MAX_TOKENS" \
    --request-timeout "$REQUEST_TIMEOUT" \
    --port "$UI_PORT" &
UI_PID=$!
set +m

stopping=""

shutdown_ui() {
    [ -n "$stopping" ] && return
    stopping=1
    trap '' INT TERM HUP
    kill -0 "$UI_PID" 2>/dev/null || return
    echo
    echo "Stopping the chat UI (the server keeps running)..."
    kill -INT "$UI_PID" 2>/dev/null
    waited=0
    while kill -0 "$UI_PID" 2>/dev/null; do
        sleep 0.5
        waited=$((waited + 1))
        if [ "$waited" -ge 20 ]; then
            kill -KILL "$UI_PID" 2>/dev/null
            break
        fi
    done
}

# HUP is what Terminal sends when the window is closed; without a trap the
# Gradio process would be terminated by default action instead of shut down,
# and could survive as an orphan holding the port.
trap shutdown_ui INT TERM HUP
trap shutdown_ui EXIT

wait "$UI_PID"
