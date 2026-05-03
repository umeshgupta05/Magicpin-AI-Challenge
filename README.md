# Vera Challenge Bot

Rules-first FastAPI bot for the magicpin Vera AI Challenge. The bot makes deterministic trigger decisions from category, merchant, trigger, and optional customer context, then optionally asks a fast hosted LLM to polish wording. If the LLM is missing, rate-limited, slow, or unsafe, the rules message is returned.

## Approach

- `compose(category, merchant, trigger, customer=None)` is deterministic and grounded only in supplied context.
- `/v1/tick` ranks active triggers by urgency and business value, suppresses duplicates, and returns judge-ready actions.
- `/v1/reply` handles replay scenarios: auto-replies, explicit intent, hostile opt-outs, and off-topic replies.
- Optional LLM polish order defaults to `groq,gemini`; local Ollama can be enabled for development with `LLM_PROVIDER_ORDER=ollama`.

## Model Choice (and Why)

- **Primary behavior**: rules-first composer (no external dependency) for deterministic, judge-safe outputs.
- **Optional polish model**: Gemini 2.5 Flash Lite (default in env) for fast, low-latency copy refinement.
- **Why this split**: the rules engine selects facts and CTAs; the LLM only improves wording. This keeps responses within the 30s budget and prevents hallucinated facts from impacting scores.
- **Fallbacks**: if the LLM is slow or unavailable, responses degrade to the deterministic draft without failing the run.

## Run Locally

```bash
pip install -r requirements.txt
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Optional hosted LLM polish:

```bash
set GROQ_API_KEY=your_key
set LLM_PROVIDER_ORDER=groq,gemini
set LLM_TIMEOUT_SECONDS=7
```

No key is required for correctness because the deterministic fallback is always active.

## Test

```bash
python dataset/generate_dataset.py --seed-dir dataset --out expanded
python make_submission.py --data expanded --out submission.jsonl
```

In `judge_simulator.py`, set `BOT_URL = "http://localhost:8080"` and choose a judge LLM provider. Then run:

```bash
python judge_simulator.py
```

## Deploy (Render)

This repo includes a `start.sh` and `Procfile` for Render-style deploys.

1. Create a new Web Service from this repo.
2. Build command: `pip install -r requirements.txt`
3. Start command: `sh start.sh`
4. Set env vars (optional but recommended):
   - `TEAM_NAME`, `TEAM_MEMBERS`, `CONTACT_EMAIL`, `BOT_VERSION`, `SUBMITTED_AT`
   - `LLM_POLISH_ENABLED=0` if you do not want to use an LLM key
   - Or set `GROQ_API_KEY` / `GEMINI_API_KEY` for polish mode

The server binds to `0.0.0.0` and uses Render's `PORT` if present (defaults to 8080).

## Tradeoffs

The implementation optimizes reliability over free-form creativity:

- **Reliability > creativity**: deterministic drafts always exist; LLM is optional and only used for polish.
- **Speed > depth**: short, single-CTA messages are favored to fit the 30s judge timeout and reduce multi-turn risk.
- **Context-only grounding**: no external data calls; avoids fabrication but limits personalization to provided context.
- **Conservative safety checks**: strict validation may reject some otherwise good LLM outputs to keep outputs judge-safe.
