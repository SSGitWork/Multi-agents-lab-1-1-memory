# memory.py
# Lab 1.1 — Two-tier memory system for the Coder Agent
#
# TODO: Implement the three components marked below.
# Read the docstring for each before writing any code.
# build_context() and the Chroma setup are already done for you.

import os
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

SLIDING_WINDOW_SIZE = 6
TOP_K_MEMORIES = 3

# ── Helicone-routed client (shared with agent.py) ─────────────────────────────

_HELICONE_BASE = os.getenv("HELICONE_BASE_URL")
_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
_HELICONE_API_KEY = os.getenv("HELICONE_API_KEY")

helicone_client = OpenAI(
    # Helicone key doubles as the auth token.
    api_key=_OPENROUTER_API_KEY,
    base_url=_HELICONE_BASE,
    default_headers={
        "Helicone-Auth": f"Bearer {_HELICONE_API_KEY}",
    }
)

# ── Custom embedding function routing through Helicone ────────────────────────

class HeliconeEmbeddingFunction(EmbeddingFunction):
    """Routes all embedding calls through Helicone instead of calling OpenAI directly."""

    def __call__(self, input: Documents) -> Embeddings:
        response = helicone_client.embeddings.create(
            model="text-embedding-3-small",
            input=input
        )
        return [item.embedding for item in response.data]

# ── Chroma setup (already done for you) ──────────────────────────────────────

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="agent_memory",
    embedding_function=HeliconeEmbeddingFunction(),
    metadata={"hnsw:space": "cosine"}
)

# ── TODO 1: SlidingWindowBuffer ───────────────────────────────────────────────

class SlidingWindowBuffer:
    """
    Keeps the N most recent conversation turns in memory.

    Each turn is a dict: {"role": "user" | "assistant", "content": str}

    add() must evict the oldest turn when the buffer exceeds max_turns.
    get() must return a list of turn dicts in chronological order.
    """

    def __init__(self, max_turns: int = SLIDING_WINDOW_SIZE):
        self.max_turns = max_turns
        self._turns: list[dict] = []

    def add(self, role: str, content: str) -> None:
        if not role:
            raise ValueError("role must not be empty")
        if content is None:
            raise ValueError("content must not be None")

        self._turns.append({"role": role, "content": content})
        if len(self._turns) > self.max_turns:
            self._turns.pop(0)

    def get(self) -> list[dict]:
        return list(self._turns)

    def __len__(self) -> int:
        return len(self._turns)


# ── TODO 2: store_memory ──────────────────────────────────────────────────────

_memory_counter = 0

def store_memory(text: str, metadata: dict | None = None) -> str:
    """
    Embed and store a string in the Chroma collection.

    Generate a unique ID (e.g. mem_0001, mem_0002, ...).
    Call collection.add() with ids, documents, and metadatas.
    Return the ID string.
    """
    global _memory_counter

    if not text:
        raise ValueError("text must not be empty")

    _memory_counter += 1
    memory_id = f"mem_{_memory_counter:04d}"
    collection.add(
        ids=[memory_id],
        documents=[text],
        metadatas=[metadata or {}],
    )
    return memory_id


# ── TODO 3: retrieve_memories ─────────────────────────────────────────────────

def retrieve_memories(query: str, top_k: int = TOP_K_MEMORIES) -> list[str]:
    """
    Return the top-k most semantically similar memories for a query string.

    Guard against querying an empty collection — return [] if collection.count() == 0.
    Use collection.query() with n_results=min(top_k, collection.count()).
    Return the list of document strings (not metadata, not IDs).
    """
    if not query:
        return []

    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, count),
    )
    documents = results.get("documents", [[]])
    if not documents or not documents[0]:
        return []
    return [doc for doc in documents[0] if doc]


# ── build_context (already done for you) ─────────────────────────────────────

def build_context(query: str, window: SlidingWindowBuffer) -> list[dict]:
    """
    Assembles the full message list for an LLM call:
      1. System prompt with retrieved long-term memories injected
      2. Recent turns from the sliding window
    """
    long_term = retrieve_memories(query)

    memory_block = ""
    if long_term:
        formatted = "\n".join(f"  [{i+1}] {m}" for i, m in enumerate(long_term))
        memory_block = f"\n\nRelevant memories from earlier in this session:\n{formatted}"

    system_message = {
        "role": "system",
        "content": (
            "You are a Coder Agent. Help with coding tasks. "
            "Use the memories below if relevant to the current question."
            + memory_block
        )
    }
    return [system_message] + window.get()
