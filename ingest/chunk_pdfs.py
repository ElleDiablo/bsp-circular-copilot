"""Split every usable circular PDF into small, overlapping chunks of text.

Usage:
    uv run python -m ingest.chunk_pdfs

Reads data/pdf_manifest.csv (only status "ok") and data/pdfs/{number}.pdf.
Writes data/chunks.jsonl: one JSON object per line, one line per chunk:
    {"circular_no": "1242", "page": 3, "chunk_index": 0, "text": "..."}

No AWS needed: embedding is a separate, later step that reads this file.
"""

import csv  # standard library: read the manifest CSV row by row
import json  # standard library: turn each chunk (a Python dict) into one line of JSON text
import logging  # standard library: lets us quiet pypdf's harmless warnings
from pathlib import Path  # standard library: OS-independent paths; / joins them

from pypdf import PdfReader  # installed package: reads a PDF's text, page by page

MANIFEST = Path("data/pdf_manifest.csv")  # input: one row per circular with its status
PDF_DIR = Path("data/pdfs")  # input: the PDFs collect_pdfs copied here ({number}.pdf)
OUT = Path("data/chunks.jsonl")  # output: one chunk per line (the "silver" layer)

# Rule of thumb: about 4 characters = 1 token in English prose, but tables of numbers can be
# under 2 characters per token. Measured: at 1500 chars, 254 of 6,821 chunks went over the
# 512-token limit of mxbai-embed-large / Cohere Embed v3; at 800 chars, none did.
CHUNK_CHARS = 800  # ≈ 200 tokens of prose, still under 512 tokens for number-heavy tables
OVERLAP_CHARS = 120  # each chunk repeats the last 120 chars of the previous one, so a cut sentence survives whole

MIN_CHARS = 100 # chunks shorter than this could be page labels like "Appendix 1" so there's too little meaning to embed


logging.getLogger("pypdf").setLevel(logging.ERROR)  # only show pypdf messages that are real errors


def chunk(text: str) -> list[str]:  # takes one page's text, returns a list of chunk strings
    """Cut one page of text into overlapping pieces of at most CHUNK_CHARS."""
    text = " ".join(text.split())  # split() breaks on any whitespace; join glues back with single spaces
    pieces = []  # the chunks we'll return
    start = 0  # position (character index) where the current chunk begins
    while start < len(text):  # keep going while there's text left (an empty page makes no chunks)
        pieces.append(text[start : start + CHUNK_CHARS])  # slice: characters start .. start+799
        if start + CHUNK_CHARS >= len(text):  # this piece already reached the end of the page
            break  # so stop; otherwise we'd make a tiny chunk that's only overlap
        start += CHUNK_CHARS - OVERLAP_CHARS  # move forward 680, not 800, so chunks overlap by 120
    return pieces  # e.g. a 2000-char page → chunks starting at 0, 680, 1360


def main() -> None:  # the script's entry point; returns nothing
    with MANIFEST.open(encoding="utf-8") as f:  # open the manifest; utf-8 so characters like ñ survive
        numbers = [  # build a list of circular numbers...
            row["circular_no"]  # ...taking the number from each row...
            for row in csv.DictReader(f)  # ...where each row is a dict keyed by the header...
            if row["status"] == "ok"  # ...but only PDFs with real text (skip no_text / missing)
        ]

    total = 0  # how many chunks we wrote, for the summary
    longest = 0  # length of the longest chunk, to check we stay under the token limit
    skipped = 0  # skipped pieces with less than 100 chars per chunk
    with OUT.open("w", encoding="utf-8") as out:  # "w" overwrites each run → no duplicates (idempotent)
        for number in numbers:  # one circular at a time
            reader = PdfReader(PDF_DIR / f"{number}.pdf")  # open data/pdfs/1242.pdf (/ joins Path parts)
            for page_no, page in enumerate(reader.pages, start=1):  # page_no counts 1, 2, 3… like real pages
                text = page.extract_text() or ""  # extract_text() can return None on an empty page → use ""
                for i, piece in enumerate(chunk(text)):  # i = 0, 1, 2… position of the chunk within the page
                    if len(piece) < MIN_CHARS: # too short to mean anything
                        skipped += 1           # skip
                        continue
                    record = {  # one chunk, with everything needed to cite it later
                        "circular_no": number,  # which circular
                        "page": page_no,  # which page (for "BSP Circular No. X, p. N")
                        "chunk_index": i,  # which piece of that page
                        "text": piece,  # the chunk itself
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")  # dict → one JSON line; keep ñ readable
                    total += 1  # count it
                    longest = max(longest, len(piece))  # remember the biggest chunk we've seen

    print(f"{len(numbers)} circulars -> {total} chunks -> {OUT}")  # summary, e.g. 319 circulars -> 11813 chunks
    print(f"longest chunk: {longest} chars (~{longest // 4} tokens)")  # // is whole-number division
    print(f"skipped {skipped} chunks under {MIN_CHARS} chars")


if __name__ == "__main__":  # true only when run directly (python -m ingest.chunk_pdfs), not when imported
    main()  # so run the script
