# Ollama Bridge MCP — Give Claude Code Multiple Brains

An MCP server that connects [Claude Code](https://docs.anthropic.com/en/docs/claude-code) to [Ollama](https://ollama.com), letting Claude delegate tasks to dozens of specialized AI models — locally or via Ollama's cloud.

**The idea:** Claude (Sonnet 4.6/Opus 4.6) stays as the orchestrator — the brain that understands your codebase, manages context, and drives the workflow. But instead of doing *everything* itself, it can now call out to specialized models for specific tasks: a math model for hard reasoning, a code model for generation, a vision model for diagrams, an OCR model for documents.

## Why Multi-Model? Every Major AI Tool Already Does This

This isn't a novel concept — it's how the industry actually works. By studying the [system prompts and internals of 30+ AI coding tools](https://github.com/ElnurIbrahimov/system-prompts-and-models-of-ai-tools), a clear pattern emerges:

- **GitHub Copilot** supports 6+ backend models: Claude Sonnet 4, Gemini 2.5 Pro, GPT-4.1, GPT-4o, GPT-5, GPT-5-mini
- **Cursor** uses GPT-5 for its agent mode, different models for different features
- **Augment Code** switches between Claude Sonnet 4 and GPT-5 depending on the task
- **Windsurf/Cascade** runs on GPT 4.1, **Same.dev** on GPT-4.1, each product picking the best model for their use case
- **Perplexity's Comet** uses multiple specialized models behind its browser agent

No commercial product uses a single model for everything. They all route tasks to specialized models. The difference is: they do it behind closed doors, and you pay a premium for it.

**This MCP server gives you the same architecture — open source, on your terms.**

Claude stays as the orchestrator (the best model for understanding context and managing complex workflows), and cheap/free specialized models handle the delegated subtasks. Same pattern as the billion-dollar products, but you control the routing and pay a fraction of the cost.

## Why This Exists

### The Problem
Claude Code is powerful, but every token costs money. Using Opus for everything — including tasks a smaller specialized model could handle — burns through your API budget fast.

### The Solution
This MCP server lets Claude **delegate** to other models through Ollama:

- **Reduce Claude API usage** — Offload tasks like code generation, math, OCR, and research to cheaper or free models
- **More brains, less cost** — Ollama's free tier gives you local models at zero cost. Their [$20/month cloud plan](https://ollama.com/pricing) gives access to massive models (up to 1T parameters) with very generous usage limits
- **Specialization** — Some models beat Claude at specific tasks. Kimi K2 scores 99.1% on AIME math. MiniMax M2.5 scores 80.2% on SWE-bench. Use the right tool for the job.
- **Keep Claude as the orchestrator** — Sonnet 4.6 routes tasks and manages your session. Opus 4.6 is the heavy hammer when you need it. Everything else gets delegated.

### The Architecture

```
You <-> Claude Code (Sonnet 4.6 / Opus 4.6)
              |
              |--- orchestrates, thinks, manages context
              |
              |---> [MCP] ---> Ollama ---> Specialized Model
              |                              - code gen
              |                              - math reasoning
              |                              - vision/OCR
              |                              - second opinions
              |                              - batch comparisons
              |
              |--- comes back with the answer, continues working
```

Claude sees all available models and their strengths. It picks the right one (or several at once), sends the task, gets the result, and keeps going. You don't have to manage routing — Claude does it automatically.

## Tools

The server exposes 4 MCP tools:

| Tool | What It Does |
|------|-------------|
| **`ask_model`** | Send a prompt to any specific Ollama model |
| **`batch_ask`** | Send the same prompt to multiple models **in parallel** — get compared answers |
| **`route_task`** | Describe a task category (code, math, vision, etc.) and the server **automatically picks the best available model** |
| **`list_models`** | List all models on your Ollama instance |

### `batch_ask` — Multi-Model Comparison

Send one prompt to 2-5 models at the same time. All run in parallel, so total time = slowest model, not the sum. Use cases:

- Get 3 different code implementations and pick the best
- Cross-check a factual answer across models to reduce hallucination
- Compare reasoning approaches on a hard problem
- Get a fast local answer immediately while a thorough cloud answer is still generating

### `route_task` — Smart Auto-Routing

Don't want to pick a model? Just tell it the task type:

| Category | Description | Top Models |
|----------|-------------|------------|
| `code` | Code generation, review, debugging | devstral-2:123b-cloud, minimax-m2.5:cloud, qwen3-coder-next:cloud |
| `math` | Math, logic, proofs, formal reasoning | kimi-k2-thinking:cloud, cogito-2.1:671b-cloud |
| `reasoning` | Complex reasoning, chain-of-thought | deepseek-v3.2:cloud, cogito-2.1:671b-cloud, deepseek-r1:8b |
| `general` | Research, Q&A, analysis, summarization | kimi-k2.5:cloud, deepseek-v3.2:cloud, mistral-large-3:675b-cloud |
| `vision` | Image understanding, diagrams | qwen3-vl:235b-cloud, mistral-large-3:675b-cloud, gemma3:4b |
| `ocr` | Document parsing, tables, formulas | glm-ocr:latest |
| `factual` | Factual Q&A (low hallucination priority) | glm-5:cloud, cogito-2.1:671b-cloud |
| `agentic` | Multi-step tasks, tool use, automation | nemotron-3-nano:30b-cloud, qwen3.5:397b-cloud |
| `fast` | Quick simple tasks, speed over quality | gemini-3-flash-preview:cloud, deepcoder:1.5b, qwen2:1.5b |

The router checks which models you actually have installed and picks the best available match. No errors if you don't have the top choice — it falls back gracefully.

## Available Models (27 total: 15 cloud + 12 local)

### Cloud Models (via Ollama Cloud — $20/mo, 0 bytes stored locally)

| Model | Params | Best For | Key Benchmark |
|-------|--------|----------|---------------|
| `deepseek-v3.2:cloud` | 671B MoE | General reasoning, agents | GPT-5-level, 160K context |
| `devstral-2:123b-cloud` | 123B | Coding agent | SWE-bench 72.2%, MIT license |
| `cogito-2.1:671b-cloud` | 671B (37B active) | Frontier reasoning | Competes with closed models, MIT |
| `mistral-large-3:675b-cloud` | 675B MoE | Multimodal enterprise | Vision + 11 langs, 256K context |
| `nemotron-3-nano:30b-cloud` | 30B (3.5B active) | Agentic tasks | NVIDIA, 1M context, 3.3x throughput |
| `qwen3-coder-next:cloud` | 80B (3B active) | Efficient coding | Sonnet 4.5-level, 256K context |
| `gemini-3-flash-preview:cloud` | Google | Fast frontier | Speed-first, vision capable |
| `minimax-m2.5:cloud` | 230B (10B active) | Code + function calling | SWE-bench 80.2% |
| `kimi-k2-thinking:cloud` | 1T (32B active) | Deep math | AIME 99.1% |
| `kimi-k2.5:cloud` | 1T (32B active) | All-rounder, agent swarms | MMLU 92.0, SWE 76.8% |
| `glm-5:cloud` | 744B (40B active) | Low hallucination | SWE 77.8%, lowest hallucination |
| `qwen3.5:397b-cloud` | 397B (17B active) | Agentic browsing | BrowseComp 78.6%, 1M context |
| `qwen3-coder:480b-cloud` | 480B (35B active) | Large codebase nav | 256K-1M context |
| `qwen3-vl:235b-cloud` | 235B (22B active) | Vision — images, diagrams | MMMU 85.0 |
| `gpt-oss:120b-cloud` | 117B (5.1B active) | Fast second opinions | MMLU 90%, AIME 97.9% |

### Local Models (Free — runs on your GPU, no network needed)

All local models fit on an 8GB VRAM GPU (RTX 4060 or similar).

| Model | Size | Best For |
|-------|------|----------|
| `deepseek-r1:8b` | 5.2 GB | Reasoning, chain-of-thought (approaches O3) |
| `deepcoder:1.5b` | 1.1 GB | Code reasoning (O3-mini level), ultra-fast |
| `gemma3:4b` | 3.3 GB | Multimodal (text + images), 128K context, 140 languages |
| `qwen3:8b` | 5.2 GB | Quick general tasks |
| `glm-ocr:latest` | 2.2 GB | OCR, documents, tables, formulas |
| `qwen2.5-coder:7b` | 4.7 GB | Quick code questions |
| `mistral:7b` | 4.4 GB | Fast general tasks |
| `llama3:8b` | 4.7 GB | Fast general tasks |
| `deepseek-coder:6.7b` | 3.8 GB | Quick code tasks |
| `nomic-embed-text` | 274 MB | Text embeddings for RAG/search |
| `qwen2:1.5b` | 934 MB | Ultra-fast simple tasks |
| `llava:latest` | 4.7 GB | Local vision tasks |

You can use **any model available on Ollama** — these are just the ones with routing hints baked into the server. Pull whatever you need with `ollama pull <model>`.

## Setup

### Prerequisites

- [Ollama](https://ollama.com/download) installed and running (`ollama serve`)
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI installed
- Python 3.10+

### 1. Clone this repo

```bash
git clone https://github.com/ElnurIbrahimov/ollama-bridge-mcp.git
cd ollama-bridge-mcp
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull some models

Cloud models download nothing — they're just pointers to Ollama's cloud infrastructure. Local models download to your disk.

```bash
# Cloud models (instant, 0 bytes — requires Ollama account for $20/mo plan)
ollama pull deepseek-v3.2:cloud
ollama pull devstral-2:123b-cloud
ollama pull kimi-k2-thinking:cloud
ollama pull kimi-k2.5:cloud
ollama pull cogito-2.1:671b-cloud
ollama pull mistral-large-3:675b-cloud
ollama pull nemotron-3-nano:30b-cloud
ollama pull qwen3-coder-next:cloud
ollama pull gemini-3-flash-preview:cloud
ollama pull minimax-m2.5:cloud
ollama pull glm-5:cloud
ollama pull qwen3.5:397b-cloud
ollama pull qwen3-coder:480b-cloud
ollama pull qwen3-vl:235b-cloud
ollama pull gpt-oss:120b-cloud

# Local models (free, runs on your GPU — needs 8GB+ VRAM)
ollama pull deepseek-r1:8b        # 5.2 GB — reasoning
ollama pull deepcoder:1.5b        # 1.1 GB — code reasoning
ollama pull gemma3:4b             # 3.3 GB — multimodal (text + images)
ollama pull qwen3:8b              # 5.2 GB — general
ollama pull glm-ocr:latest        # 2.2 GB — OCR
ollama pull qwen2.5-coder:7b     # 4.7 GB — code
```

### 4. Register the MCP server with Claude Code

```bash
claude mcp add ollama-bridge -- python /full/path/to/ollama-bridge-mcp/server.py
```

Replace `/full/path/to/` with your actual path. On Windows:

```bash
claude mcp add ollama-bridge -- python C:/Users/yourname/ollama-bridge-mcp/server.py
```

### 5. Restart Claude Code

Start a new Claude Code session. You should see `ollama-bridge` in your available tools. Claude will now automatically delegate to Ollama models when appropriate.

## Example Flows

### Single model delegation
1. You ask Claude to optimize a complex algorithm
2. Claude sends the code to `minimax-m2.5:cloud` (SWE-bench 80.2%) via `ask_model`
3. Gets back optimized code, reviews it, integrates into your codebase

### Batch comparison
1. You ask Claude to solve a tricky math proof
2. Claude uses `batch_ask` with `[kimi-k2-thinking:cloud, glm-5:cloud, gpt-oss:120b-cloud]`
3. Gets 3 independent solutions in parallel, compares them, picks the best

### Auto-routed task
1. You ask Claude to parse a PDF table
2. Claude calls `route_task` with category `ocr`
3. Server checks your installed models, picks `glm-ocr:latest`, runs it
4. Claude gets the parsed table and continues

## Cost Breakdown

| Component | Cost | What You Get |
|-----------|------|-------------|
| **Ollama Free** | $0 | Local models only (qwen3:8b, llama3:8b, etc.) |
| **Ollama Cloud** | $20/mo | Access to 1T-param models with very generous rate limits |
| **Claude Code** | Your existing plan | Sonnet 4.6 as orchestrator, Opus 4.6 for heavy lifting |

The $20/mo Ollama cloud plan is an incredible value — you get access to models with up to 1 trillion parameters, and the usage limits are very generous for development work. Most tasks won't even need cloud models; local 7-8B models handle quick questions instantly.

**Net effect:** Your Claude API usage goes down because routine tasks get offloaded to Ollama. Claude focuses on what it's best at — orchestration, context management, and the hard stuff.

## Customization

### Change the routing guide

Edit the `MODEL_GUIDE` string in `server.py` to add/remove models or change routing recommendations. Claude reads this guide to decide which model to use.

### Edit the routing table

The `ROUTING_TABLE` dict at the top of `server.py` controls `route_task` behavior. Add categories or change model priority:

```python
ROUTING_TABLE = {
    "code": {
        "models": ["devstral-2:123b-cloud", "minimax-m2.5:cloud", "qwen3-coder-next:cloud",
                    "qwen3-coder:480b-cloud", "deepcoder:1.5b", "qwen2.5-coder:7b"],
        "description": "Code generation, review, refactoring, debugging"
    },
    # Add your own categories...
    "creative": {
        "models": ["kimi-k2.5:cloud", "deepseek-v3.2:cloud", "qwen3:8b"],
        "description": "Creative writing, brainstorming, ideation"
    }
}
```

Models are tried in order — first available match wins.

### Add your own models

Pull any model from the [Ollama library](https://ollama.com/library):

```bash
ollama pull <model-name>
```

It's immediately available through the MCP server. Add it to the `MODEL_GUIDE` and `ROUTING_TABLE` if you want Claude to prioritize it for certain tasks.

### Change the Ollama endpoint

If Ollama runs on a different host/port, edit `OLLAMA_BASE` in `server.py`:

```python
OLLAMA_BASE = "http://your-host:11434"
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Cannot connect to Ollama" | Make sure Ollama is running: `ollama serve` |
| Model timeout (>600s) | Try a smaller model or simpler prompt |
| MCP server not showing up | Re-run `claude mcp add` and restart Claude Code |
| Cloud models not working | Check your Ollama account and cloud access at [ollama.com](https://ollama.com) |
| `route_task` says no models available | Pull at least one model per category you want to use |

## Related Reading

- [System Prompts and Models of AI Tools](https://github.com/ElnurIbrahimov/system-prompts-and-models-of-ai-tools) — Leaked system prompts from 30+ AI coding tools, showing how every major product uses multi-model architectures internally
- [Ollama](https://ollama.com) — Run open-source LLMs locally or via cloud
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) — Anthropic's CLI for Claude
- [MCP (Model Context Protocol)](https://modelcontextprotocol.io) — The protocol that makes this integration possible

## License

MIT
