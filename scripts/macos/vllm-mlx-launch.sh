#!/bin/bash
# Starts the vllm-mlx server and chat UI in two Terminal windows, waits for each
# to actually be ready, then opens Safari on the UI.
#
# Ordering matters: the UI queries /health at startup to check the server is
# serving a multimodal model, and opening Safari before Gradio is listening
# gives a "can't connect" page. So each stage waits for the previous one.

SERVER_PORT="${VLLM_MLX_PORT:-8000}"
UI_PORT="${VLLM_MLX_UI_PORT:-7860}"
VENV="${VLLM_MLX_VENV:-$HOME/vllm-mlx-env}"
BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVER_TIMEOUT=900   # weights load in ~60s; allow for a cold download
UI_TIMEOUT=180

notify() { osascript -e "display notification \"$2\" with title \"$1\"" >/dev/null 2>&1; }

fail() {
    osascript -e "display dialog \"$1\" with title \"vllm-mlx launcher\" buttons {\"OK\"} default button 1 with icon stop" >/dev/null 2>&1
    exit 1
}

open_terminal() {   # $1 = script path to run in a new Terminal window
    osascript >/dev/null 2>&1 <<APPLESCRIPT
tell application "Terminal"
    activate
    do script "'$1'"
end tell
APPLESCRIPT
}

server_ready() {
    curl -fs --max-time 2 "http://127.0.0.1:$SERVER_PORT/health" 2>/dev/null \
        | grep -q '"model_loaded":true'
}

ui_ready() {
    curl -fs -o /dev/null --max-time 2 "http://127.0.0.1:$UI_PORT/" 2>/dev/null
}

# --- preflight ---------------------------------------------------------------

[ -x "$VENV/bin/vllm-mlx" ] || fail "vllm-mlx is not installed in $VENV.

Install it with:
  python3 -m venv ~/vllm-mlx-env
  source ~/vllm-mlx-env/bin/activate
  pip install vllm-mlx"

[ -x "$BIN/vllm-mlx-serve.command" ] || fail "Missing $BIN/vllm-mlx-serve.command"
[ -x "$BIN/vllm-mlx-chat.command" ]  || fail "Missing $BIN/vllm-mlx-chat.command"

# --- server ------------------------------------------------------------------

if server_ready; then
    notify "vllm-mlx" "Server already running on port $SERVER_PORT"
else
    notify "vllm-mlx" "Starting server — loading ~16 GB, about a minute"
    open_terminal "$BIN/vllm-mlx-serve.command"

    waited=0
    until server_ready; do
        sleep 3
        waited=$((waited + 3))
        if [ "$waited" -ge "$SERVER_TIMEOUT" ]; then
            fail "The server did not become ready within $((SERVER_TIMEOUT / 60)) minutes.

Check the server Terminal window for the actual error — a port already in use and a failed model download look nothing alike."
        fi
    done
    notify "vllm-mlx" "Server ready on port $SERVER_PORT"
fi

# --- chat UI -----------------------------------------------------------------

if ui_ready; then
    notify "vllm-mlx" "Chat UI already running on port $UI_PORT"
else
    open_terminal "$BIN/vllm-mlx-chat.command"

    waited=0
    until ui_ready; do
        sleep 2
        waited=$((waited + 2))
        if [ "$waited" -ge "$UI_TIMEOUT" ]; then
            fail "The chat UI did not come up on port $UI_PORT within $UI_TIMEOUT seconds.

Check its Terminal window — if port $UI_PORT is taken, set VLLM_MLX_UI_PORT to a free one."
        fi
    done
fi

# --- browser -----------------------------------------------------------------

open -a Safari "http://127.0.0.1:$UI_PORT/"
notify "vllm-mlx" "Ready — chat open at 127.0.0.1:$UI_PORT"
