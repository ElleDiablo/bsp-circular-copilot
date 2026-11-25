"""Embed every chunk with Ollama and load it into Postgres (pgvector).

Usage:
    uv run --env-file .env python -m ingest.load_chunks

Reads data/circulars.csv and data/chunks.jsonl (made by chunk_pdfs.py).
    1. Upserts one row per circular into `circulars` (chunks need their parent row: foreign key).
    2. Embeds the chunks in batches with Ollama (mxbai-embed-large, 1024 numbers per chunk)
       and inserts each chunk + its vector into `chunks`.

Safe to re-run: chunks already in the table are skipped, so it resumes where it stopped.
Needs Ollama running (`brew services start ollama`) and DATABASE_URL in .env (Supabase).
"""

import csv  # standard library: reads CSV files row by row as dictionaries
import json  # standard library: turns JSON text into Python dicts/lists and back
import os  # standard library: os.environ holds environment variables like DATABASE_URL
import urllib.error  # standard library: HTTPError, raised when a server answers with an error code
import urllib.request  # standard library: sends HTTP requests (here, to Ollama); no package needed
from pathlib import Path  # standard library: file paths as objects, work on any OS

import psycopg  # third-party Postgres driver: connect to Supabase, run SQL, commit

CIRCULARS = Path("data/circulars.csv")  # input: catalog with title, date and URL per circular
CHUNKS = Path("data/chunks.jsonl")  # input: one chunk per line; this file is the silver layer

OLLAMA_URL = "http://localhost:11434/api/embed"  # Ollama's "text in, vectors out" endpoint on this Mac
MODEL = "mxbai-embed-large"  # embedding model: 1024 numbers per text, at most 512 tokens per text
BATCH = 32  # chunks per request: one request with 32 texts is much faster than 32 requests


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


def embed_batch(batch: list[dict], truncated: list[str]) -> list[list[float]]:  # chunks in, one vector per chunk out
    """Embed a batch of chunks. If one is too long, find it, cut only that one, and record it."""
    try:  # first try the fast way: the whole batch in one request
        return embed([c["text"] for c in batch])  # the usual case: all 32 fit, 32 vectors back
    except urllib.error.HTTPError:  # Ollama refused the batch: at least one chunk is over 512 tokens,
        # but the error doesn't say which one, so we retry the chunks one by one to find it.
        vectors = []  # the vectors for this batch, built up one at a time, in the same order as batch
        for c in batch:  # go through the batch's chunks one by one
            try:  # try this one chunk on its own, still with truncate=False
                vectors.append(embed([c["text"]])[0])  # it fits: embed() returns a list of 1 vector, [0] takes it
            except urllib.error.HTTPError:  # this chunk is the one over 512 tokens
                vectors.append(embed([c["text"]], truncate=True)[0])  # embed only its first 512 tokens
                truncated.append(f"{c['circular_no']} p.{c['page']} #{c['chunk_index']}")  # note which one, e.g. "984 p.41 #0"
        return vectors  # one vector per chunk, so the batch still lines up with its vectors


def to_pgvector(vector: list[float]) -> str:  # Python list in, text out
    """Write a vector the way pgvector reads it as text: '[0.01,0.03,...]'."""
    return "[" + ",".join(str(x) for x in vector) + "]"  # each number → text, joined by commas, in brackets


def main() -> None:  # the script's entry point; returns nothing
    with CHUNKS.open(encoding="utf-8") as f:  # open chunks.jsonl; "with" closes it afterwards
        chunks = [json.loads(line) for line in f]  # each line → a dict; all 11,813 fit in memory (~10 MB)

    needed = {c["circular_no"] for c in chunks}  # set of circular numbers that have chunks (319, not all 346)
    with CIRCULARS.open(encoding="utf-8") as f:  # open the catalog CSV
        circulars = [row for row in csv.DictReader(f) if row["circular_no"] in needed]  # keep only those 319 rows

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:  # connect to Supabase; "with" closes it even on errors
        # Step 1: parent rows first, because chunks.circular_no must point to an existing circular.
        conn.cursor().executemany(  # run the same SQL once per row in the list below
            """
            -- upsert: insert the row, or update it if this circular_no is already there
            insert into circulars (circular_no, title, issued_on, source_url)  -- the 4 columns we fill
            values (%s, %s, %s, %s)                       -- placeholders: psycopg fills them in safely
            on conflict (circular_no) do update           -- row exists (re-run)? update instead of failing
                set title = excluded.title,               -- excluded = the row we just tried to insert
                    issued_on = excluded.issued_on,       -- keep the date in sync with the CSV
                    source_url = excluded.source_url      -- keep the URL in sync with the CSV
            """,  # end of the SQL text
            [  # the values: one tuple per circular, in the same order as the placeholders
                (r["circular_no"], r["title"], r["issued_on"] or None, r["source_url"])  # empty date "" → None → SQL NULL
                for r in circulars  # one tuple for each of the 319 circulars
            ],  # end of the values list
        )  # end of executemany
        conn.commit()  # save the circulars now, before the long embedding loop
        print(f"circulars upserted: {len(circulars)}")  # expect 319

        # Step 2: which chunks are already in the table? Skip those (this makes it resumable).
        done = set(conn.execute("select circular_no, page, chunk_index from chunks").fetchall())  # every loaded chunk's key, as tuples
        todo = [c for c in chunks if (c["circular_no"], c["page"], c["chunk_index"]) not in done]  # chunks not loaded yet
        print(f"chunks: {len(chunks)} total, {len(done)} already loaded, {len(todo)} to embed")  # first run: 0 already loaded

        # Step 3: embed and insert, one batch at a time.
        truncated = []  # chunks over 512 tokens (OCR junk, dot leaders); filled in by embed_batch
        for start in range(0, len(todo), BATCH):  # start = 0, 32, 64, ... up to the end of todo
            batch = todo[start : start + BATCH]  # the next 32 chunks (fewer for the last batch)
            vectors = embed_batch(batch, truncated)  # one vector per chunk, in the same order
            conn.cursor().executemany(  # insert every chunk of the batch with its vector
                """
                insert into chunks (circular_no, page, chunk_index, text, embedding)  -- id fills itself (bigserial)
                values (%s, %s, %s, %s, %s::vector)               -- ::vector turns '[0.01,...]' text into a vector
                on conflict (circular_no, page, chunk_index) do nothing  -- already there? skip, don't fail
                """,  # end of the SQL text
                [  # the values: one tuple per chunk
                    (c["circular_no"], c["page"], c["chunk_index"], c["text"], to_pgvector(v))  # the 5 columns
                    for c, v in zip(batch, vectors)  # zip pairs chunk 1 with vector 1, chunk 2 with vector 2, ...
                ],  # end of the values list
            )  # end of executemany
            conn.commit()  # save this batch: if the run stops later, these rows are kept
            print(f"{start + len(batch)} / {len(todo)}", end="\r")  # progress; \r returns to the line start, so it overwrites itself

    print(f"\ndone: {len(todo)} chunks embedded and loaded")  # \n first: move off the progress line
    print(f"truncated to 512 tokens: {len(truncated)} {truncated}")  # expect 6 on a full run


if __name__ == "__main__":  # true only when run as a script (python -m ...), not when imported
    main()  # so run main()
