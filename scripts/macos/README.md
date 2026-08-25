# macOS launcher

Starts the vllm-mlx server and the Gradio chat UI in two Terminal windows, waits for
each to actually be ready, then opens the browser.

```bash
cp *.command *.sh ~/bin/          # or anywhere on your machine
chmod +x ~/bin/vllm-mlx-*
~/bin/vllm-mlx-launch.sh
```

To get a double-clickable Desktop icon, compile a one-line AppleScript that calls the
launcher detached, so the app quits immediately instead of spinning while the weights
load:

```bash
cat > /tmp/launcher.applescript <<'APPLESCRIPT'
set launcher to (POSIX path of (path to home folder)) & "bin/vllm-mlx-launch.sh"
do shell script quoted form of launcher & " > /dev/null 2>&1 &"
APPLESCRIPT
osacompile -o "$HOME/Desktop/vllm-mlx Chat.app" /tmp/launcher.applescript
```

macOS will ask once for permission to control Terminal.

## Why the sequencing matters

The chat UI queries `/health` at startup to check the server is serving a multimodal
model, and a browser opened before Gradio is listening shows a connection error. So
`vllm-mlx-launch.sh` waits for `/health` before starting the UI, and for the UI's
port before opening Safari. Re-running when things are already up is safe: it detects
both and just opens the browser.

## Shutdown

`Ctrl+C`, or closing the window, runs the same graceful path in both wrappers: SIGINT
to the child, 20s grace, SIGKILL only if it is still alive. Closing the window cannot
leave an orphan holding the port.

Stopping the **server** also stops the chat UI (`VLLM_MLX_STOP_UI=0` opts out).
Stopping the **UI** deliberately leaves the server running — it is the half that costs
a minute and 16 GB to start.

## Settings

Every value is an environment variable, so switching models is one variable rather
than an edit:

| Variable | Default |
|---|---|
| `VLLM_MLX_MODEL` | `mlx-community/Qwen3.8-27B-4bit` |
| `VLLM_MLX_SERVED_NAME` | `qwen` |
| `VLLM_MLX_PORT` | `8000` |
| `VLLM_MLX_UI_PORT` | `7860` |
| `VLLM_MLX_MAX_TOKENS` | `8192` |
| `VLLM_MLX_VENV` | `~/vllm-mlx-env` |
| `VLLM_MLX_STOP_UI` | `1` |
