# "I don't see any video" — Six Bugs Between a 27B VLM and an iPhone Clip

## How a single token count separated a model failure from four layers of plumbing failure, and what a Qwen VLM actually sees when you hand it a video

---

I asked a 27-billion-parameter vision-language model, running locally on a Mac, to describe an 8-second video of someone playing a guitar. It replied:

> I don't see any video or attachment in this conversation, so I can't describe its objects or colors.

HTTP 200. No errors in the log. A fluent, confident, entirely reasonable answer.

The model was telling the truth. The video had been accepted, base64-decoded, parsed into a `video_url` content part — and then silently discarded before it ever reached the model. Fixing that turned up five more bugs stacked behind it, each of which also returned HTTP 200 and a plausible-sounding answer.

This is a walkthrough of all six, the one measurement that found every one of them, and two things I learned about what these models actually receive when you feed them a video. Everything here is reproducible; the branch is linked at the end.

**The stack:** vllm-mlx 0.4.1, mlx-vlm 0.6.15, OpenCV 5.0.0, Python 3.14, macOS on Apple Silicon, serving `mlx-community/Qwen3.8-27B-4bit`.

---

## The measurement that finds all of them

Start here, because it makes everything else cheap.

When a multimodal request goes wrong, the failure is almost always indistinguishable from the model just being bad at the task. The answer comes back well-formed. Nothing throws. You end up tuning prompts against a pipeline that dropped your attachment three layers ago.

There is one number that separates the two cases and requires no ground truth:

**Send the same prompt twice — once with the attachment, once without — and compare `prompt_tokens`.**

```
text only:            prompt_tokens = 68
text + 22 MB video:   prompt_tokens = 68
```

Identical. Byte-identical replies, too. A video that reaches the model costs *thousands* of prompt tokens; this one cost zero. That single comparison converts "the model can't see it" into "the model never got it," and it took ten seconds.

Every bug below was found or confirmed with this check. If you take one thing from this article, take this one.

*(Credit where due: this diagnostic came from a parallel debugging session on llama.cpp, which hit the same class of failure from a completely different direction.)*

---

## Bug 1: A capability allow-list that guessed from the filename

The server log had already said it, one line, an hour before anyone read it carefully:

```
SimpleEngine loaded: mlx-community/Qwen3.8-27B-4bit (MLLM=False)
```

`MLLM=False`. The model had loaded as a **text-only LLM**. Multimodal content parts were being flattened out of every request as a matter of course.

Here is the detection logic that decided that (`vllm_mlx/api/utils.py`):

```python
def is_mllm_model(model_name: str) -> bool:
    config = _try_read_config_json(model_name)   # local DIRECTORIES only
    if config is not None:
        return _config_indicates_vlm(config)
    return _check_legacy_string_patterns(model_name)   # substring match on the NAME
```

`_try_read_config_json` returns `None` unless the argument resolves to a local directory. A HuggingFace repo ID is not a directory. So detection fell through to matching the *name* against a hardcoded list — `"-VL-"`, `"llava"`, `"gemma3"`, `"qwen3_5"`, and so on.

`"mlx-community/Qwen3.8-27B-4bit"` matches none of them.

Meanwhile the model's own `config.json` says, unambiguously:

```json
{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "image_token_id": 248056,
  "video_token_id": 248057,
  "vision_config": { "model_type": "qwen3_5" }
}
```

And that file was sitting in the local HuggingFace cache the entire time. One call away:

```python
from huggingface_hub import try_to_load_from_cache
try_to_load_from_cache(repo_id, "config.json")   # no network, returns the path
```

**The principle: prefer the declared capability over the inferred one.** The fix reads `config.json` for repo IDs as well as local paths, and demotes name-pattern matching to what it should always have been — the fallback for a model that hasn't been downloaded yet.

This matters more than one model. Auditing every model in one machine's HF cache, six of twenty-five flipped from `False` to `True`, and every one was genuinely multimodal:

| Model | Architecture |
|---|---|
| Qwen3.8-27B-4bit | `Qwen3_5ForConditionalGeneration` |
| Qwen3.6-27B-4bit | `Qwen3_5ForConditionalGeneration` |
| Qwen3.6-35B-A3B-4bit | `Qwen3_5MoeForConditionalGeneration` |
| Qwen3.6-35B-A3B-OptiQ-4bit | `Qwen3_5MoeForConditionalGeneration` |
| Qwen3-Omni-30B-A3B-Instruct-4bit | `Qwen3OmniMoeForConditionalGeneration` |
| Muse-Glimmer-30B-4bit | `MuseGlimmerForConditionalGeneration` |

Nothing flipped the other way; no text-only model carries a stray vision key. Six models mis-served on one laptop, all of them recent. Naming conventions drift; `config.json` doesn't.

---

## Bug 2: The upstream module had been deleted

With detection fixed, the same request went from a silent lie to a loud error:

```
ModuleNotFoundError: No module named 'mlx_vlm.video_generate'
```

HTTP 500. That is *progress* — a wrong answer replaced by an honest failure — but it isn't working video.

vllm-mlx's native video path imported two things from `mlx_vlm.video_generate`. In mlx-vlm 0.6.x that module no longer exists; the layout is `mlx_vlm.generate.*` now, and `process_vision_info` was removed entirely. Any model with a `video_token_id` — every Qwen3-VL-class model — took this path and hit the wall.

The port:

```python
from mlx_vlm.utils import load_video          # decode
from mlx_vlm.generate import generate         # generate(video=[...])

frames, sample_fps = load_video(
    path, fps=video_fps, max_frames=video_max_frames,
    min_frames=MIN_FRAMES, frame_factor=FRAME_FACTOR,
)
```

One deliberate choice: decode in vllm-mlx rather than handing `generate()` a path. Two reasons. First, `video_max_frames` gets honored — upstream's default cap is **768 frames**, and a five-minute clip at 2 fps would try 600 frames, roughly 450,000 prompt tokens. Second, having the frames in hand means they can be *validated* before they reach the model (see Bug 3).

If the native path is unavailable, the request now falls back to frames-as-images instead of raising. That fallback is a working substitute, not an equivalent one — stills go through the image encoder rather than the model's temporal 3D convolution, so token cost per frame, effective resolution, and motion tracking all differ. It logs a warning saying exactly that, because two paths that produce different stimuli should never be silently interchanged.

Result: **68 → 12,110 prompt tokens**, and a correct description of the clip.

---

## Bug 3: Everything that could fail quietly

Same session, three more places where a media failure produced no signal:

**The frame sampler returned whatever it got.** Frame-index seeking can silently skip frames on long-GOP codecs like HEVC, and an empty list was a perfectly valid return value — the request then ran with no visual input at all. Now:

```python
def assert_video_decoded(frames, video_path, *, height_axis=0, width_axis=1):
    if len(frames) == 0:
        raise ValueError(
            f"Decoded 0 frames from {video_path}. The container or codec could "
            "not be read - this is a decode failure, not a model failure."
        )
    ...
```

That last clause is the point. The error message names the layer, so the next person doesn't spend an afternoon blaming the model. It also logs the successful case — `Video decode OK: 16 frames, 2160x2604` — which makes frame count and orientation visible in the server log instead of merely inferable.

**The multimodal processor caught every media exception and continued.** `logger.warning` and carry on, producing a request with no media and a confident answer about nothing. Now it raises. A 4xx or 5xx naming the real cause is worth far more than a plausible 200.

**The temp-file suffix was built by string-splitting the MIME type.** `mime.split("/")[-1]` turns `video/quicktime` into `.quicktime` and `video/x-matroska` into `.x-matroska`. FFmpeg content-sniffs, so OpenCV opened these anyway — I verified `.quicktime`, `.mov`, and no suffix at all decode identically. Latent, not live. Fixed regardless, with a real MIME→container map, because several decoders (PyAV, decord) infer the container from the extension and this would become live the moment anyone swaps the backend.

---

## Bug 4: The UI could show you a chip for a file it had thrown away

Three changes in the Gradio chat app, all aimed at the same silence:

Unknown file extensions **defaulted to `image/jpeg`**. A mislabeled video is rejected or misread server-side, and the guess hides the real problem. It now resolves by extension, then by the system MIME database, then raises. The tables gained `.mov` / `video/quicktime` — what an iPhone produces and what browsers report for it — along with `.m4v`, `.mkv`, `.heic`.

The **`prompt_tokens` check from the top of this article is now a product feature**: if attachments were sent and the server reports an implausibly low prompt token count, the reply carries a visible warning naming the likely cause and how to check it.

And **startup queries `/health`**, warning when the server has loaded a text-only model while the UI is in multimodal mode — the exact configuration of Bug 1, caught before the first message instead of after twenty.

---

## Bug 5: A flag that worked everywhere except where you needed it

Video worked. Video was also *slow* — and then it stopped working again:

```
Error: 504 Server Error: Gateway Timeout
```

The model is a reasoner. Started with thinking disabled server-wide:

```
--default-chat-template-kwargs '{"enable_thinking": false}'
```

Text replies came back in under a second. Image replies, instant. Video requests reasoned for 8,192 tokens — about eight minutes at 17 tok/s — and blew through the server's 300-second request timeout.

The flag was working. It just never reached the video path. The native branch built its chat-template arguments from `tools` alone and dropped `enable_thinking` on the floor. Confirmed straight against the model's own template:

```
enable_thinking=True   ->  ...<|im_start|>assistant\n<think>\n
enable_thinking=False  ->  ...<|im_start|>assistant\n<think>\n\n</think>\n\n
```

A pre-closed think block versus an open one. That one dropped keyword was the difference between no reasoning at all and reasoning until the token budget ran out.

This is the same species as Bug 1 and Bug 4, and it's the reason it stayed hidden: **a setting appears to work because the paths that honor it are the ones you test first.** Text works, images work, so the flag is fine — and the one path that ignores it is the one you were actually trying to use.

After the fix: **396 tokens in 60.8 seconds, HTTP 200.**

---

## What we did *not* port, and why that mattered most

The parallel llama.cpp session had fixed two genuine video bugs and offered them for porting:

1. **The moov atom.** MP4/MOV keep their index in a `moov` atom usually written at the *end* of the file. Decoding needs a backward seek to find it. Pipe the bytes to FFmpeg and that seek fails: zero frames, no exception, HTTP 200.
2. **Rotation metadata.** iPhones record landscape and attach a `rotation=-90` display matrix. The FFmpeg CLI *applies* the rotation and emits 2160×2604; `ffprobe` *reports* the stored 2604×2160. Read dimensions from one and buffers from the other and you parse the image at the wrong stride — and because a transpose preserves the byte count exactly, nothing errors. The picture just shears into diagonal bands, which the model then dutifully describes as "horizontal color bands with scanline texture."

Both are real. **Neither applied here, and porting the second one would have broken a correct pipeline.**

vllm-mlx decodes with `cv2.VideoCapture` on a *path*, never a pipe, so there is no moov problem. And OpenCV 5.0.0 auto-rotates:

```python
cap = cv2.VideoCapture("IMG_2537.mov")   # stored 2604x2160, rotation=-90
cap.get(cv2.CAP_PROP_ORIENTATION_AUTO)   # 1.0
ret, frame = cap.read()
frame.shape                              # (2604, 2160, 3) — height-major, upright
```

Copying llama.cpp's width/height swap into that would have *introduced* the exact shear it was written to remove.

Two rules came out of this, and they generalize past this project:

**Test before you patch.** A fix is written against a specific decoder's behavior. The same fix applied to a decoder that behaves differently is a new bug wearing a trusted commit message.

**Look at the frame.** Not the shape, not the byte count — dump it to PNG and open it. I did, saw a seafoam Stratocaster with a sun-motif strap, and knew in five seconds that the rotation path was fine. No amount of documentation-reading would have settled it as fast, and reading the docs would have given the wrong answer, because the OpenCV auto-rotation default is exactly the sort of thing that changes between versions.

---

## What the model actually sees

With the pipeline fixed, two surprises about the stimulus itself. Both are standard behavior that nobody tells you about, and both change how you read a model's answer.

### It sees temporal *pairs*, not frames

The model kept describing "four frames," "seven images," "a crossfade," "double exposure," "重影" (ghosting). Its own frame counts were never consistent.

Asking a model how many images it received is not evidence. So build a stimulus where the count is *observable*: an 8-second clip at 24 fps with a large number burned in, changing every half second, 1 through 16.

```python
import cv2, numpy as np
W, H, FPS, SECS, n_labels = 720, 900, 24, 8, 16
out = cv2.VideoWriter('counter.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
total = FPS * SECS
for i in range(total):
    label = i // (total // n_labels) + 1
    img = np.full((H, W, 3), 245, np.uint8)
    cv2.putText(img, str(label), (W//2-190, H//2+130), cv2.FONT_HERSHEY_SIMPLEX,
                9.0, (20, 20, 20), 30, cv2.LINE_AA)
    out.write(img)
out.release()
```

Sixteen frames sampled at 2 fps. The model's answer:

> 2, 3, 5, **90**, 12, 13, 16; 7 images

Look at `90`. There is no frame labelled 90. That is **9 and 10 superimposed** — one fused pair, caught in the act. The other numbers are pair representatives. Drop to four frames and it collapses further:

> the image shows two digits, 1 and 6

The mechanism is one line in the model's `video_preprocessor_config.json`:

```json
{ "temporal_patch_size": 2, "merge_size": 2, "patch_size": 16 }
```

The vision encoder's 3D convolution fuses **each adjacent pair of frames** into a single temporal slot. Sixteen frames arrive as eight slots. The model isn't confused about your video — it is accurately describing what its own encoder hands it. Every "double exposure" and "crossfade" report was that pairing, and every frame count it gave was a count of slots, undercounted.

### Frames are nearly free; resolution is the budget

The obvious assumption is that doubling the frame rate doubles the cost. Measured across both clips:

| `video_fps` | frames | guitar clip, 2604×2160 | counter clip, 720×900 |
|---|---|---|---|
| 2.0 | 16 | 12,095 | 5,025 |
| 1.0 | 8 | 12,055 | 2,561 |
| 0.5 | 4 | 11,071 | 1,329 |

The counter clip scales roughly linearly. The guitar clip is **flat**.

`max_pixels: 25165824` is a budget for the *entire video*, not per frame. A high-resolution clip saturates it at any frame count, so fewer frames simply buys more resolution per frame — 756 tokens/frame at 16 frames, 2,768 at 4. The low-resolution clip never reaches the ceiling, so it scales with frames.

Practically: on phone-resolution video, **more frames costs you almost nothing in tokens and costs you spatial detail instead.** When the model hedged between a Telecaster and a Stratocaster on a clip that plainly shows a double-cutaway Strat body, the fix wasn't a better prompt — it was `"video_fps": 0.5`, trading motion for four frames at nearly four times the detail each.

---

## Reproduce it

```bash
git clone https://github.com/sbayer2/vllm-mlx.git
cd vllm-mlx
git checkout fix/video-silently-dropped
```

Six commits, each self-contained:

```
8d174bf  Stop replaying historical media in Gradio chat
59e9110  fix(video): honor enable_thinking on the native video path
d921030  fix(chat-ui): surface attachments the server dropped
1a0d5a5  fix(video): port the native video path to the mlx-vlm 0.6 API
c554d3e  fix(video): fail loudly on a video that did not decode
57c95f7  fix(detection): read config.json for HF repo IDs in is_mllm_model
```

Serving:

```bash
source ~/vllm-mlx-env/bin/activate

vllm-mlx serve mlx-community/Qwen3.8-27B-4bit \
  --served-model-name qwen --port 8000 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --timeout 900
```

And the chat UI, in a second terminal:

```bash
source ~/vllm-mlx-env/bin/activate

vllm-mlx-chat --server-url http://127.0.0.1:8000 \
  --served-model-name qwen --max-tokens 8192
```

`--served-model-name` must match on both, or every reply is `Error: The model 'default' does not exist`.

The A/B that started all of this, which you can point at any multimodal server:

```python
import base64, requests

URL = "http://localhost:8000/v1/chat/completions"
PROMPT = "Describe exactly what you see in this video."

def ask(content):
    r = requests.post(URL, json={"model": "qwen", "max_tokens": 1,
                                 "messages": [{"role": "user", "content": content}]})
    return r.json()["usage"]["prompt_tokens"]

b64 = base64.b64encode(open("clip.mov", "rb").read()).decode()
print("text only:", ask(PROMPT))
print("with video:", ask([
    {"type": "text", "text": PROMPT},
    {"type": "video_url", "video_url": {"url": f"data:video/quicktime;base64,{b64}"}},
]))
```

Two numbers within a few dozen of each other means your attachment is being dropped, whatever the model says in its answer.

---

## Still open

Two things I have not resolved, stated plainly rather than left for someone to trip over:

**A shutdown segfault after video requests.** Exiting the server cleanly after processing video produces a segfault during interpreter teardown, after `Application shutdown complete`. A text-only model exits cleanly (status 0); the same 27B VLM with only text requests also exits cleanly. So it correlates with the video path, but I never ran the decisive test — 27B, one video request, then SIGINT with a fault handler attached. It is post-shutdown, so nothing in flight is at risk, but it is unexplained.

**Per-video sampling rate with multiple clips.** When a request carries more than one video, the first clip's sampled fps is applied to all of them, because `mlx_vlm.prepare_inputs` takes a single scalar `fps`. That value drives Qwen's interleaved timestamp tokens, so the second clip's frames get labelled using the first clip's rate. Small when the sources have similar frame rates; wrong in principle. Fixing it means calling the processor directly instead of routing through `mlx_vlm.generate`.

---

## The pattern

Five of the six bugs shared one shape: **an allow-list, a lookup, or a keyword that didn't recognize valid input, and no error anywhere.**

- A MIME allow-list without `video/quicktime` (llama.cpp, same week)
- A capability check that matched on names instead of reading the declared config
- An import against a module that upstream had renamed
- A file-type resolver that guessed `image/jpeg` for anything it didn't know
- A template keyword forwarded on three paths out of four

Every one returned HTTP 200 with a coherent answer. Not a single one raised. And in every case, the model got blamed for a plumbing failure — which is the expensive part, because you can burn days tuning prompts against a pipeline that discarded your input before inference began.

Multimodal pipelines have many places to drop a payload and very few places that complain. So assert on the artifact, not on the absence of an exception. Count the frames. Look at the frame. And when in doubt, send the same prompt twice and diff the token count.

---

# Appendix: The five-minute version

Everything above is the investigation. This is the part you can act on without reading any of it.

**You're in the right place if** you're running an MLX vision model behind a chat UI — LM Studio, SGLang, Open WebUI, llama.cpp's server, vllm-mlx's own — and either your `.mov` files vanish or get rejected, or a single video takes minutes to come back.

Two independent problems, two fixes. Neither requires patching your server.

---

## Step 0: Install it

Apple Silicon Mac, macOS with Homebrew. `ffmpeg` is needed for the conversion in Fix 1.

```bash
brew install python@3.12 ffmpeg

python3 -m venv ~/vllm-mlx-env
source ~/vllm-mlx-env/bin/activate
pip install --upgrade pip
```

Then install vllm-mlx. **Pick one:**

```bash
# (a) The PyPI release — does NOT contain the video fixes in this article
pip install vllm-mlx

# (b) The patched branch — video actually works
pip install "vllm-mlx @ git+https://github.com/sbayer2/vllm-mlx.git@fix/video-silently-dropped"
```

Everything else comes along automatically: `mlx`, `mlx-vlm`, `opencv-python`, `transformers`, `gradio`, `fastapi` are all hard dependencies, so there are no extras to remember. Python 3.10 or newer is required.

Verify:

```bash
pip show vllm-mlx | head -2
vllm-mlx --help
```

**A note on hardware.** `Qwen3.8-27B-4bit` is roughly a 15 GB download and holds about 16 GB resident once loaded, so it wants 32 GB of unified memory to be comfortable and 64 GB to leave room for anything else. On a smaller machine, use the model from vllm-mlx's own docs instead — `mlx-community/Qwen3-VL-4B-Instruct-3bit` — and substitute it everywhere below. The first `serve` downloads the weights automatically; subsequent starts read from the HuggingFace cache and take about a minute.

---

## Step 1: Run it

**Terminal 1 — the server:**

```bash
source ~/vllm-mlx-env/bin/activate

vllm-mlx serve mlx-community/Qwen3.8-27B-4bit \
  --served-model-name qwen --port 8000 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --timeout 900
```

Wait for `Uvicorn running on http://127.0.0.1:8000`. Roughly a minute to load ~16 GB.

**Terminal 2 — the chat UI:**

```bash
source ~/vllm-mlx-env/bin/activate

vllm-mlx-chat --server-url http://127.0.0.1:8000 \
  --served-model-name qwen --max-tokens 8192
```

Then open **http://127.0.0.1:7860**.

Three things that will bite you if you skip them:

- **`--served-model-name` must match on both commands.** Mismatch it and every reply is `Error: The model 'default' does not exist. Available model: 'qwen'`.
- **`--default-chat-template-kwargs '{"enable_thinking": false}'`** is what keeps a video reply near a minute instead of eight (Fix 3).
- **Type a question alongside the attachment.** An empty textbox sends the clip with no text part, and the model — having no language cue — is liable to answer in Chinese.

Two commands worth knowing when something looks wrong:

```bash
curl -s localhost:8000/health          # model_type should say "mllm", not "llm"
curl -s localhost:8000/v1/models       # the id here is your --served-model-name
```

---

## Fix 1: Convert the clip once. It solves four things at once.

```bash
ffmpeg -i IMG_2537.mov \
  -vf scale=-2:720 \
  -c:v libx264 -pix_fmt yuv420p \
  -movflags +faststart \
  -an \
  clip.mp4
```

On the test clip: **16.7 MB → 1.25 MB**, and `ffprobe` now reports `598x720`, `h264`, with no rotation side data at all. Each flag is doing specific work:

| flag | what it fixes | section above |
|---|---|---|
| output `.mp4` + `libx264` | UI MIME allow-lists that omit `video/quicktime`, and decoders without HEVC | Bug 4 |
| `-movflags +faststart` | moves the `moov` atom to the **front** of the file, so decoders reading from a pipe or a stream can find the index without seeking backwards | "What we did not port" |
| `-vf scale=-2:720` | physically applies the rotation and drops the display matrix, so a decoder that ignores rotation metadata can't shear the frame — and cuts the token cost (Fix 2) | "What we did not port" |
| `-an` | drops the audio track, which non-omni models ignore anyway |  |

If your only problem is the container and you want to keep full resolution, remux losslessly instead — no re-encode, near-instant:

```bash
ffmpeg -i IMG_2537.mov -c copy -movflags +faststart clip.mp4
```

Note this keeps the HEVC video and the rotation metadata, so it fixes the MIME and moov problems but not the codec or rotation ones.

---

## Fix 2: Understand what you're paying for, then pay less

For Qwen3-VL-class models the prompt cost has a closed form:

```
prompt_tokens  ≈  total_pixels ÷ 2048
                  where total_pixels = frames × width × height
                  capped at max_pixels ÷ 2048
```

The `÷ 2048` is `patch_size² × merge_size² × temporal_patch_size` = `16² × 2² × 2`. The cap comes from your model's `video_preprocessor_config.json` — for Qwen3.8-27B, `max_pixels: 25165824`, so **12,288 tokens is the ceiling no matter what you send**.

Measured on the same server, same model, same 16 frames:

| clip | resolution | pixels | predicted | measured | wall time |
|---|---|---|---|---|---|
| original `.mov` | 2604×2160 | 90M → capped at 25.2M | 12,288 | 12,110 | 60.8 s |
| converted `.mp4` | 598×720 | 6.9M | 3,364 | **3,359** | **10.6 s** |

Prediction and measurement agree to within 5 tokens. And the wall time is **6× faster** for the same 16 frames of the same video — because video latency here is dominated by prefill, not generation.

This is the practical consequence of the pixel-budget finding above: a 4K phone clip saturates the budget and gets *silently downscaled by the processor anyway*. You are not gaining detail by sending it — you're paying full prefill for pixels that get thrown away. Downscale it yourself and choose where the savings go.

Rules of thumb for a 16-frame sample:

- Below about **1.57 MP per frame** (~1620×970) you're under the cap and paying only for what you send.
- Above it you're at the ceiling regardless, so sending a bigger file buys nothing.
- Want fine detail instead of motion? Send **fewer frames** (`video_fps: 0.5`), not a bigger file. Four frames at the same budget get ~2,768 tokens each instead of ~756.

---

## Fix 3: Stop the model reasoning for eight minutes

Qwen3.5/3.8 are reasoning models. Left alone, a video request will happily spend its entire `max_tokens` budget inside `<think>` before writing a word of answer — at 17 tok/s that's about eight minutes for an 8,192-token budget, and on many servers it trips a 300-second gateway timeout first.

Three ways to stop it, in order of preference:

**Server-wide**, if you control the launch command:

```bash
vllm-mlx serve mlx-community/Qwen3.8-27B-4bit \
  --served-model-name qwen --port 8000 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --timeout 900
```

**Per request**, which works from any UI that lets you add fields to the JSON body:

```json
{
  "model": "qwen",
  "messages": [ ... ],
  "chat_template_kwargs": { "enable_thinking": false }
}
```

Measured on the converted clip: 3,366 prompt tokens, **40 completion tokens, 10.6 seconds**, and a correct one-sentence answer naming the double-cutaway body, three single-coil pickups and white pickguard.

**Capped**, if your UI can do neither and you just want a ceiling on the reasoning:

```bash
--default-thinking-token-budget 256
```

Two traps worth knowing:

- **`/no_think` in the prompt does not work for this model.** That convention comes from earlier Qwen releases; the string appears zero times in Qwen3.8's chat template. It will be read as literal text.
- **On vllm-mlx you need the Bug 5 fix for any of this to apply to *video*.** Before that commit, `enable_thinking` reached text and image requests and was dropped on the video path — so the flag looks like it's working right up until you attach a clip.

---

## Fix 4: Confirm it before you tune anything

Whatever server you're on, run the A/B from the "Reproduce it" section above before you touch a prompt. It is the same ten-second check that found every bug in this article, and it works against any OpenAI-compatible endpoint.

What healthy looks like, on the converted clip:

```
prompt_tokens for text alone     ~50-70
prompt_tokens with the video     3,000-12,000
```

And in the server log, if you're on the patched vllm-mlx:

```
Video decode OK: 16 frames, 598x720, from /var/folders/.../tmpXXXX.mp4
Native video: 1 video(s), 16 frames total, sampled at 1.96 fps
```

Two things to read off that line: the **frame count** should be non-zero and roughly `fps × duration`, and the **dimensions** should be the *display* orientation — portrait video should report a height larger than its width. If the numbers are transposed, you have the rotation bug and the frames are being read at the wrong stride.

---

## If you're on LM Studio, SGLang, or llama.cpp

The fixes in this article are vllm-mlx commits, but the failure modes aren't vllm-mlx-specific — they're what happens when a container format, a capability check and a decoder disagree. On a server you don't control, check in this order:

1. **Does the UI accept the file at all?** If a `.mov` disappears from the composer with no error, its allow-list is missing `video/quicktime`. Fix 1 sidesteps it entirely.
2. **Does the request carry the video?** The A/B check. If prompt tokens don't move, stop debugging the model.
3. **Does the decoder get a path or a bytestream?** Anything piping bytes to FFmpeg needs `+faststart` — which Fix 1 already did for you.
4. **Are the frames upright?** Decode one frame, write it to PNG, and look at it. Byte counts survive a transpose; your eyes don't.

That fourth one is worth repeating, because it's the cheapest test in this whole article and the one that settled the biggest question in it.

---

*Fixes: [github.com/sbayer2/vllm-mlx](https://github.com/sbayer2/vllm-mlx), branch `fix/video-silently-dropped`, against [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx).*
