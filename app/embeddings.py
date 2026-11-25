"""The one place that defines the embedding model, used by both ingest and search.

Questions and chunks must be embedded by the SAME model: vectors from different models
live on different "maps" of meaning and can't be compared. Keeping the model name here,
once, means the loader and the search can never drift apart.
"""

import json  # standard library: turns Python dicts into JSON text and back
import urllib.error  # standard library: HTTPError, raised when Ollama answers with an error code
import urllib.request  # standard library: sends HTTP requests to Ollama; no package needed

OLLAMA_URL = "http://localhost:11434/api/embed"  # Ollama's "text in, vectors out" endpoint on this Mac
MODEL = "mxbai-embed-large"  # embedding model: 1024 numbers per text, at most 512 tokens per text

# mxbai-embed-large was trained to see this phrase in front of QUESTIONS (not in front of chunks).
# It tells the model "this is a search query", which slightly improves matching (~+0.02 measured).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed(texts: list[str], truncate: bool = False) -> list[list[float]]:  # list of texts in, list of vectors out
    """Send a list of texts to Ollama, get back one vector (1024 numbers) per text, same order."""
    # truncate=False (the default here): a text over 512 tokens makes Ollama answer with an
    # error (HTTP 400) instead of quietly embedding only its first 512 tokens.
    body = json.dumps({"model": MODEL, "input": texts, "truncate": truncate}).encode()  # dict → JSON text → bytes (HTTP sends bytes)
    request = urllib.request.Request(  # build the request (not sent yet)
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}  # having data makes it a POST; header says "this is JSON"
    )  # end of the request
    with urllib.request.urlopen(request, timeout=300) as response:  # send it; wait up to 300 s for the reply
        return json.load(response)["embeddings"]  # reply is {"embeddings": [[...1024...], ...]}: return that list


def embed_query(question: str) -> list[float]:  # one question in, one vector out
    """Embed a user's question for search (with the query prefix the model expects)."""
    return embed([QUERY_PREFIX + question])[0]  # embed() takes a list, so wrap it; [0] takes the one vector back


def to_pgvector(vector: list[float]) -> str:  # Python list in, text out
    """Write a vector the way pgvector reads it as text: '[0.01,0.03,...]'."""
    return "[" + ",".join(str(x) for x in vector) + "]"  # each number → text, joined by commas, in brackets
