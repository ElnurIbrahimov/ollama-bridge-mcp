# Ollama Bridge MCP — Give Claude Code Multiple Brains

An MCP server that connects [Claude Code](https://docs.anthropic.com/en/docs/claude-code) to [Ollama](https://ollama.com), letting Claude delegate tasks to dozens of specialized AI models — locally or via Ollama's cloud.

**The idea:** Claude (Sonnet 4.6/Opus 4.6) stays as the orchestrator — the brain that understands your codebase, manages context, and drives the workflow. But instead of doing *everything* itself, it can now call out to specialized models for specific tasks: a math model for hard reasoning, a code model for generation, a vision model for diagrams, an OCR model for documents.

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
              |
              |--- comes back with the answer, continues working
```

Claude sees all available models and their strengths. It picks the right one, sends the task, gets the result, and keeps going. You don't have to manage routing — Claude does it automatically.

## Available Models

### Cloud Models (via Ollama Cloud — $20/mo)

| Model | Params | Best For | Key Benchmark |
|-------|--------|----------|---------------|
| `minimax-m2.5:cloud` | 230B (10B active) | Code generation, function calling | SWE-bench 80.2% |
| `kimi-k2-thinking:cloud` | 1T (32B active) | Deep math, chain-of-thought | AIME 99.1% |
| `kimi-k2.5:cloud` | 1T (32B active) | All-rounder, agent swarms | MMLU 92.0, SWE 76.8% |
| `glm-5:cloud` | 744B (40B active) | Low hallucination, reliable | SWE 77.8%, lowest hallucination |
| `qwen3.5:397b-cloud` | 397B (17B active) | Agentic browsing, tools | BrowseComp 78.6%, 1M context |
| `qwen3-coder:480b-cloud` | 480B (35B active) | Large codebase nav | 256K-1M context window |
| `qwen3-vl:235b-cloud` | 235B (22B active) | Vision — images, diagrams | MMMU 85.0 |
| `gpt-oss:120b-cloud` | 117B (5.1B active) | Fast second opinions | MMLU 90%, AIME 97.9% |

### Local Models (Free — runs on your hardware)

| Model | Size | Best For |
|-------|------|----------|
| `qwen3:8b` | 8B | Quick general tasks |
| `glm-ocr:latest` | 0.9B | OCR, documents, tables, formulas |
| `qwen2.5-coder:7b` | 7B | Quick code questions |
| `mistral:7b` | 7B | Fast general tasks |
| `llama3:8b` | 8B | Fast general tasks |
| `deepseek-coder:6.7b` | 6.7B | Quick code tasks |
| `nomic-embed-text` | 137M | Text embeddings for RAG/search |
| `qwen2:1.5b` | 1.5B | Ultra-fast simple tasks |
| `llava:latest` | 7B | Local vision tasks |

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

```bash
# Local models (free, runs on your GPU/CPU)
ollama pull qwen3:8b
ollama pull qwen2.5-coder:7b
ollama pull glm-ocr:latest

# Cloud models (requires Ollama account)
ollama pull minimax-m2.5:cloud
ollama pull kimi-k2-thinking:cloud
ollama pull kimi-k2.5:cloud
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

## How It Works

The server exposes two MCP tools:

| Tool | Description |
|------|-------------|
| `ask_model` | Send a prompt to any Ollama model with an optional system prompt |
| `list_models` | List all models currently available on your Ollama instance |

The `ask_model` tool description includes a **routing guide** — Claude reads this and automatically picks the best model for each subtask. You can customize the routing guide in `server.py` to match whatever models you have available.

### Example Flow

1. You ask Claude Code to solve a complex math problem in your codebase
2. Claude recognizes this is heavy reasoning and calls `ask_model` with `kimi-k2-thinking:cloud`
3. Kimi K2 returns the solution
4. Claude integrates the answer and continues working on your code

All transparent. You see it happen in the Claude Code output.

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

### Add your own models

Pull any model from the [Ollama library](https://ollama.com/library):

```bash
ollama pull <model-name>
```

It's immediately available through the MCP server. Add it to the `MODEL_GUIDE` if you want Claude to prioritize it for certain tasks.

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

## License

MIT
