"""The one place that defines the embedding model, used by both ingest and search.

Questions and chunks must be embedded by the SAME model: vectors from different models
live on different "maps" of meaning and can't be compared. Keeping the model name here,
once, means the loader and the search can never drift apart.

"""

import json  # standard library: turns Python dicts into JSON text and back
import os
import urllib.error  # standard library: HTTPError, raised when Ollama answers with an error code
import urllib.request  # standard library: sends HTTP requests to Ollama; no package needed

# Where Ollama lives. On the Mac: localhost. Inside a container, "localhost" is the container
# itself, so `docker run` passes OLLAMA_HOST=http://host.docker.internal:11434 (= the Mac).
# ChatOllama reads the same variable by itself, so one setting moves chat AND embeddings.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")  # env var if set, else this Mac
OLLAMA_URL = f"{OLLAMA_HOST}/api/embed"  # Ollama's "text in, vectors out" endpoint
MODEL = "mxbai-embed-large"  # embedding model: 1024 numbers per text, at most 512 tokens per text

# mxbai-embed-large was trained to see this phrase in front of QUESTIONS (not in front of chunks).
# It tells the model "this is a search query", which slightly improves matching (~+0.02 measured).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# What engine runs mxbai: "ollama" on the Mac, "fastembed" inside the container (No Ollama in cloud)
# same model, same vectors tested in experiments/embed_match_text.py (cosine 1.0000)
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "ollama") 
FASTEMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"   # fastembed's ONNX copy of mxbai-embed-large
_fastembed = None   # loaded model; created on first use, then reused (loading takes seconds)

def _embed_fastembed(texts: list[str]) -> list[list[float]]: # same contract as embed()
    """Embed in-process with fastembed (ONNX on the CPU), no server needed"""
    global _fastembed           # "global": assign to the module-level variable not local 
    if _fastembed is None:      # first call: load model (from FASTEMBED_CACHE_PATH, baked into the image)
        from fastembed import (
            TextEmbedding,  # imported here, so MAC and CI never load it
        )
        
        _fastembed = TextEmbedding(FASTEMBED_MODEL) 
    return [vector.tolist() for vector in _fastembed.embed(texts)] # numpy array -> plain list


def embed(texts: list[str], truncate: bool = False) -> list[list[float]]:  # list of texts in, list of vectors out
    """Send a list of texts to Ollama, get back one vector (1024 numbers) per text, same order."""
    if EMBED_PROVIDER == "fastembed":       # in container : no Ollama to call
        return _embed_fastembed(texts)  
    
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
