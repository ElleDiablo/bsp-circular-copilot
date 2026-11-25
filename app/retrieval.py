"""Find the chunks closest in meaning to a question, each with its citation.

Usage (from the terminal, to try it out):
    uv run --env-file .env python -m app.retrieval "What is the minimum capital for rural banks?"

question → embed with the same model as the chunks → pgvector finds the TOP_K nearest chunks
→ drop matches below MIN_SCORE → passages with "BSP Circular No. X, p. N".
An empty result means "no source": later, the agent refuses instead of guessing.
"""

import os  # standard library: os.environ holds DATABASE_URL (loaded from .env by uv --env-file)
import sys  # standard library: sys.argv is the list of words typed after the command
from dataclasses import dataclass  # standard library: auto-writes __init__ for data

import psycopg  # third-party Postgres driver: connect to Supabase, run SQL

from app.embeddings import embed_query, to_pgvector  # same model as the loader

TOP_K = 6  # how many chunks to return at most: enough context, still cheap for the chat model
# Similarity = 1 - cosine distance (1.0 = same meaning). Measured on 2026-11-25: on-topic
# questions scored 0.66-0.82, off-topic ones (banana bread, NBA, France) 0.40-0.48.
# 0.55 sits in the gap. Re-check with the real eval questions (Day 4).
MIN_SCORE = 0.55


@dataclass  # makes Passage(...) work with these fields, plus a readable print, with no extra code
class Passage:  # one search result: a chunk plus what we need to cite it
    circular_no: str  # e.g. "1165"
    title: str  # the circular's title, from the circulars table
    page: int  # page number in the PDF (1-based), so the citation is exact
    text: str  # the chunk's text, which the chat model will read
    score: float  # similarity to the question, 0..1 (higher = closer in meaning)

    def cite(self) -> str:  # the citation string the answer will show
        return f"BSP Circular No. {self.circular_no}, p. {self.page}"  # e.g. "BSP Circular No. 1165, p. 1"


def search(question: str, k: int = TOP_K, min_score: float = MIN_SCORE) -> list[Passage]:  # question in, passages out
    """Return up to k chunks closest to the question, dropping those below min_score."""
    vector = to_pgvector(embed_query(question))  # the question as '[0.01,...]' text, ready for SQL
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:  # connect to Supabase; "with" closes it after
        rows = conn.execute(  # run one query, with the values below filled into the %(name)s placeholders
            """
            select ch.circular_no, c.title, ch.page, ch.text,          -- what we need for a Passage
                   1 - (ch.embedding <=> %(v)s::vector) as score       -- <=> = cosine distance; 1 - it = similarity
            from chunks ch                                              -- ch = short name for the chunks table
            join circulars c using (circular_no)                        -- add each chunk's circular (for the title)
            order by ch.embedding <=> %(v)s::vector                     -- nearest first (this uses the HNSW index)
            limit %(k)s                                                 -- only the top k
            """,  # end of the SQL text
            {"v": vector, "k": k},  # named values: %(v)s gets the vector, %(k)s gets k
        ).fetchall()  # get all result rows as a list of tuples
    passages = [Passage(*row) for row in rows]  # *row spreads the 5 columns into Passage's 5 fields, in order
    return [p for p in passages if p.score >= min_score]  # keep only real matches; may be empty


def main() -> None:  # lets you try search() from the terminal
    question = " ".join(sys.argv[1:]) or "What is the minimum capital for rural banks?"  # words after the command, or a default
    passages = search(question)  # run the search
    print(f"Q: {question}")  # echo the question
    if not passages:  # nothing passed MIN_SCORE
        print(f"No source found (no chunk scored >= {MIN_SCORE}). The agent would refuse.")  # the "refuse" path
    for p in passages:  # one line per passage, best first
        print(f"{p.score:.2f}  {p.cite()}  {p.text[:100]}...")  # score, citation, first 100 characters


if __name__ == "__main__":  # true only when run as a script (python -m app.retrieval), not when imported
    main()  # so run main()
