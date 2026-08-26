#!/bin/bash
# vllm-mlx inference server. Double-clickable, or launched by vllm-mlx-launch.sh.
#
# Shutdown handling:
#   Ctrl+C, or closing this window, runs the same graceful path — SIGINT to the
#   server (uvicorn's own graceful shutdown), then SIGKILL only if it is still
#   alive 20s later. Closing the window therefore cannot leave a 16 GB orphan
#   holding port 8000, which is the failure that produces "Address already in
#   use" on the next launch.

MODEL="${VLLM_MLX_MODEL:-mlx-community/Qwen3.8-27B-4bit}"
SERVED_NAME="${VLLM_MLX_SERVED_NAME:-qwen}"
PORT="${VLLM_MLX_PORT:-8000}"
# 65536 prompt tokens costs ~4.3 GB of KV cache on this model: only 16 of its 64
# layers are full attention, the other 48 are linear and keep fixed-size state.
# The model itself allows 262144. Prefill is the real cost — a 12k-token video
# prompt already takes ~50s, so a 65k one takes minutes. This raises the ceiling,
# it does not make large prompts fast.
MAX_REQUEST_TOKENS="${VLLM_MLX_MAX_REQUEST_TOKENS:-65536}"
VENV="${VLLM_MLX_VENV:-$HOME/vllm-mlx-env}"
GRACE_SECONDS=20

source "$VENV/bin/activate" || {
    echo "Could not activate $VENV — is vllm-mlx installed there?"
    read -r -p "Press return to close. "
    exit 1
}

echo "Model:  $MODEL"
echo "Served: $SERVED_NAME on port $PORT"
echo "Thinking ON at ${VLLM_MLX_REASONING_EFFORT:-medium} effort — the model"
echo "reasons before answering, so replies are slower than with it off."
echo "Loading ~16 GB, roughly a minute."
echo "Ctrl+C or closing this window stops the server cleanly."
echo

# Job control puts the server in its OWN process group, so a Ctrl+C aimed at
# this window reaches only this script. The trap below then forwards it
# deliberately — without this, the server would get the terminal's SIGINT AND
# ours, and uvicorn reads a second SIGINT as "force quit, skip cleanup".
set -m

# Reasoning is ON at medium effort. The template's default when thinking is
# enabled is 'xhigh', which prepends "think carefully, validate key
# assumptions, consider plausible alternatives" and can burn the whole
# max_tokens budget before answering. 'medium' adds no such instruction.
# Valid values are xhigh, medium, low — anything else makes the template
# raise, and every request fails.
# --reasoning-parser splits <think>...</think> out of the reply into a
# separate reasoning_content field, so the answer arrives clean. The model
# still reasons and still spends the tokens; only the packaging changes.
# --timeout 900 so a slow prefill can't be cut off by the 300s default.
vllm-mlx serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --default-chat-template-kwargs \
      "{\"enable_thinking\": ${VLLM_MLX_THINKING:-true}, \"reasoning_effort\": \"${VLLM_MLX_REASONING_EFFORT:-medium}\"}" \
    --reasoning-parser "${VLLM_MLX_REASONING_PARSER:-qwen3}" \
    --max-request-tokens "$MAX_REQUEST_TOKENS" \
    --max-tokens "${VLLM_MLX_SERVER_MAX_TOKENS:-32768}" \
    --timeout 900 &
SERVER_PID=$!
set +m

stopping=""

shutdown_server() {
    [ -n "$stopping" ] && return
    stopping=1
    trap '' INT TERM HUP

    # No early return when the server is already gone: it may have been
    # stopped from outside this window (a relaunch, or a kill by pid), and the
    # chat UI still needs stopping in that case.
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo
        echo "Stopping the server..."
        kill -INT "$SERVER_PID" 2>/dev/null

        waited=0
        while kill -0 "$SERVER_PID" 2>/dev/null; do
            sleep 0.5
            waited=$((waited + 1))
            if [ "$waited" -ge $((GRACE_SECONDS * 2)) ]; then
                echo "Still running after ${GRACE_SECONDS}s — forcing."
                kill -KILL "$SERVER_PID" 2>/dev/null
                break
            fi
        done
    fi

    stop_chat_ui
}

stop_chat_ui() {
    # The launcher starts both windows together, so stopping the server stops
    # the UI with it. Matching on the server URL keeps this precise: a chat UI
    # pointed at some other server is left alone. The reverse is deliberately
    # NOT true — closing the UI leaves the server up, because the server is the
    # half that costs a minute and 16 GB to start.
    [ "${VLLM_MLX_STOP_UI:-1}" = "0" ] && return
    if pkill -INT -f "vllm-mlx-chat .*--server-url http://127.0.0.1:$PORT" 2>/dev/null; then
        echo "Chat UI stopped."
    fi
}

# HUP is what Terminal sends when the window is closed.
trap shutdown_server INT TERM HUP
trap shutdown_server EXIT

wait "$SERVER_PID"
status=$?

# 139 = 128 + SIGSEGV. vllm-mlx can crash during interpreter teardown AFTER
# "Application shutdown complete" — i.e. once the server has already stopped
# serving and every request is finished. Undiagnosed, but harmless; say so
# rather than leaving the shell's bare "segmentation fault" as the last word.
if [ "$status" -eq 139 ]; then
    echo
    echo "Server stopped. (It crashed during Python's teardown, after shutdown"
    echo "completed — a known issue that does not affect served requests.)"
elif [ "$status" -eq 130 ] || [ "$status" -eq 0 ]; then
    echo
    echo "Server stopped cleanly."
else
    echo
    echo "Server exited with status $status."
fi
