import asyncio
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server("ollama-bridge")

OLLAMA_BASE = "http://localhost:11434"

# Task categories for smart routing
ROUTING_TABLE = {
    "code": {
        "models": ["minimax-m2.5:cloud", "qwen3-coder:480b-cloud", "qwen2.5-coder:7b"],
        "description": "Code generation, review, refactoring, debugging"
    },
    "math": {
        "models": ["kimi-k2-thinking:cloud", "gpt-oss:120b-cloud", "qwen3:8b"],
        "description": "Math, logic, formal reasoning, proofs"
    },
    "general": {
        "models": ["kimi-k2.5:cloud", "glm-5:cloud", "qwen3:8b"],
        "description": "General research, Q&A, analysis, summarization"
    },
    "vision": {
        "models": ["qwen3-vl:235b-cloud", "llava:latest"],
        "description": "Image understanding, diagrams, visual analysis"
    },
    "ocr": {
        "models": ["glm-ocr:latest"],
        "description": "Document parsing, OCR, tables, formulas"
    },
    "factual": {
        "models": ["glm-5:cloud", "kimi-k2.5:cloud", "qwen3:8b"],
        "description": "Factual questions where low hallucination matters"
    },
    "fast": {
        "models": ["qwen3:8b", "qwen2:1.5b", "mistral:7b"],
        "description": "Quick simple tasks where speed matters most"
    }
}

MODEL_GUIDE = """ROUTING GUIDE — pick the right model for the task:

=== RECOMMENDED ROUTING ===
Code generation/review  -> minimax-m2.5:cloud   (80.2% SWE-bench, best function calling)
Deep math/reasoning     -> kimi-k2-thinking:cloud (99.1% AIME, chain-of-thought)
General research/all-round -> kimi-k2.5:cloud    (92.0 MMLU, 76.8% SWE-bench, agent swarms)
Factual/low-hallucination  -> glm-5:cloud        (77.8% SWE-bench, industry-lowest hallucination)
Agentic browsing/tools  -> qwen3.5:397b-cloud    (78.6% BrowseComp, 1M context, 17B active)
Heavy code generation   -> qwen3-coder:480b-cloud (69.6% SWE-bench, 256K-1M context)
Vision/images/diagrams  -> qwen3-vl:235b-cloud   (best open VL model, 85.0 MMMU)
Second opinion/general  -> gpt-oss:120b-cloud    (90% MMLU, fast, 5.1B active)
Quick local tasks       -> qwen3:8b              (instant, no network)
OCR/documents           -> glm-ocr:latest        (94.6 OmniDocBench, 0.9B, local)

=== CLOUD MODELS (detailed) ===
- minimax-m2.5:cloud      | 230B/10B active  | SWE 80.2% | BFCL 76.8 | Best coder + function caller
- kimi-k2-thinking:cloud  | 1T/32B active    | SWE 71.3% | AIME 99.1% | Math god, 200+ sequential tool calls
- kimi-k2.5:cloud         | 1T/32B active    | SWE 76.8% | MMLU 92.0 | Best all-rounder, 100 sub-agents
- glm-5:cloud             | 744B/40B active  | SWE 77.8% | AIME 92.7% | Lowest hallucination rate
- qwen3.5:397b-cloud      | 397B/17B active  | AIME 91.3 | BrowseComp 78.6% | Efficient agentic model
- qwen3-coder:480b-cloud  | 480B/35B active  | SWE 69.6% | 256K-1M context | Large codebase navigation
- qwen3-vl:235b-cloud     | 235B/22B active  | MMMU 85.0 | Vision-language SOTA | Images, video, diagrams
- gpt-oss:120b-cloud      | 117B/5.1B active | MMLU 90%  | AIME 97.9% | Fast open reasoning

=== LOCAL MODELS (instant, no latency) ===
- qwen3:8b               | 8B   | Quick general tasks
- glm-ocr:latest          | 0.9B | OCR, document parsing, tables, formulas (94.6 OmniDocBench)
- qwen2.5-coder:7b        | 7B   | Quick code questions
- mistral:7b              | 7B   | Fast general tasks
- llama3:8b               | 8B   | Fast general tasks
- deepseek-coder:6.7b     | 6.7B | Quick code tasks
- nomic-embed-text:latest  | 137M | Text embeddings for RAG/search
- qwen2:1.5b              | 1.5B | Ultra-fast simple tasks
- llava:latest             | 7B   | Local vision tasks

=== NOTES ===
- MiniMax M2.5 has benchmark gaming history — use GLM-5 or Kimi K2.5 if reliability matters more
- Kimi K2 Thinking is slower due to chain-of-thought but more thorough
- Cloud models stream from Ollama cloud — fast, no local GPU needed
- Local models run on your hardware — instant but limited capability"""


async def _query_model(client, model, messages):
    """Send a chat request to a single Ollama model and return formatted result."""
    try:
        resp = await client.post(
            f"{OLLAMA_BASE}/api/chat",
            json={"model": model, "messages": messages, "stream": False}
        )
        if resp.status_code != 200:
            return model, None, f"Ollama error {resp.status_code}: {resp.text}"

        result = resp.json()
        content = result["message"]["content"]
        total_ns = result.get("total_duration", 0)
        total_s = round(total_ns / 1e9, 1) if total_ns else "?"
        return model, total_s, content

    except httpx.TimeoutException:
        return model, None, "TIMEOUT (>600s)"
    except httpx.ConnectError:
        return model, None, "CONNECTION FAILED — is Ollama running?"
    except Exception as e:
        return model, None, f"Error: {e}"


@app.list_tools()
async def list_tools():
    categories = "\n".join(
        f"  {k}: {v['description']} (models: {', '.join(v['models'][:2])}...)"
        for k, v in ROUTING_TABLE.items()
    )

    return [
        Tool(
            name="ask_model",
            description=(
                "Send a prompt to any Ollama model (local or cloud). "
                "Use this to delegate tasks to specialized models.\n\n"
                + MODEL_GUIDE
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Ollama model name exactly as listed (e.g. 'qwen3-coder:480b-cloud')"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The prompt/question to send to the model"
                    },
                    "system": {
                        "type": "string",
                        "description": "Optional system prompt to set context/role",
                        "default": ""
                    }
                },
                "required": ["model", "prompt"]
            }
        ),
        Tool(
            name="batch_ask",
            description=(
                "Send the SAME prompt to MULTIPLE models concurrently and get all responses. "
                "Use this to get second opinions, compare approaches, or validate answers across models. "
                "All models run in parallel — total time equals the slowest model, not the sum.\n\n"
                "Example use cases:\n"
                "- Get 3 different code implementations and pick the best\n"
                "- Cross-check a factual answer across models to reduce hallucination\n"
                "- Compare reasoning approaches for a hard problem\n"
                "- Get a fast local answer while waiting for a thorough cloud answer"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of Ollama model names to query (e.g. ['kimi-k2.5:cloud', 'glm-5:cloud', 'qwen3:8b'])"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The prompt/question to send to all models"
                    },
                    "system": {
                        "type": "string",
                        "description": "Optional system prompt applied to all models",
                        "default": ""
                    }
                },
                "required": ["models", "prompt"]
            }
        ),
        Tool(
            name="route_task",
            description=(
                "Automatically pick the best model for a task category. "
                "Describe the category and this tool returns the best available model "
                "from your Ollama instance, then queries it.\n\n"
                "Categories:\n" + categories + "\n\n"
                "Use this when you know WHAT kind of task it is but don't want to "
                "pick a specific model. The router checks which models you actually "
                "have available and picks the best match."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": list(ROUTING_TABLE.keys()),
                        "description": "Task category: code, math, general, vision, ocr, factual, fast"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The prompt/question to send"
                    },
                    "system": {
                        "type": "string",
                        "description": "Optional system prompt",
                        "default": ""
                    }
                },
                "required": ["category", "prompt"]
            }
        ),
        Tool(
            name="list_models",
            description="List all available Ollama models with their strengths and recommended use cases.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):

    if name == "list_models":
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{OLLAMA_BASE}/api/tags")
                models = resp.json()["models"]
                names = [m["name"] for m in models]
                online = "\n".join(f"  - {n}" for n in names)
                return [TextContent(
                    type="text",
                    text=f"Connected to Ollama. Models available:\n{online}\n\n{MODEL_GUIDE}"
                )]
        except Exception as e:
            return [TextContent(type="text", text=f"Error connecting to Ollama: {e}")]

    elif name == "ask_model":
        model = arguments["model"]
        prompt = arguments["prompt"]
        system = arguments.get("system", "")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=600.0) as client:
            model_name, duration, content = await _query_model(client, model, messages)

        if duration is None:
            return [TextContent(type="text", text=f"[{model_name}] FAILED: {content}")]

        return [TextContent(
            type="text",
            text=f"[{model_name}] ({duration}s):\n\n{content}"
        )]

    elif name == "batch_ask":
        models = arguments["models"]
        prompt = arguments["prompt"]
        system = arguments.get("system", "")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=600.0) as client:
            tasks = [_query_model(client, m, messages) for m in models]
            results = await asyncio.gather(*tasks)

        parts = []
        for model_name, duration, content in results:
            if duration is None:
                parts.append(f"### [{model_name}] FAILED\n{content}")
            else:
                parts.append(f"### [{model_name}] ({duration}s)\n{content}")

        header = f"Batch results from {len(models)} models:\n\n"
        return [TextContent(type="text", text=header + "\n\n---\n\n".join(parts))]

    elif name == "route_task":
        category = arguments["category"]
        prompt = arguments["prompt"]
        system = arguments.get("system", "")

        candidates = ROUTING_TABLE.get(category, ROUTING_TABLE["general"])["models"]

        # Check which candidates are actually available
        available = []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{OLLAMA_BASE}/api/tags")
                installed = {m["name"] for m in resp.json()["models"]}
                available = [m for m in candidates if m in installed]
        except Exception:
            available = []

        if not available:
            return [TextContent(
                type="text",
                text=f"No models available for category '{category}'. "
                     f"Recommended models: {', '.join(candidates)}. "
                     f"Pull one with: ollama pull {candidates[0]}"
            )]

        chosen = available[0]

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=600.0) as client:
            model_name, duration, content = await _query_model(client, chosen, messages)

        if duration is None:
            return [TextContent(type="text", text=f"[{model_name}] (routed: {category}) FAILED: {content}")]

        return [TextContent(
            type="text",
            text=f"[{model_name}] (routed: {category}, {duration}s):\n\n{content}"
        )]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream, write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
