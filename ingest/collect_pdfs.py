"""Copy downloaded circular PDFs into data/pdfs/, check each one, write a checklist.

Usage:
    uv run python -m ingest.collect_pdfs ~/Downloads

For every circular in data/circulars.csv, looks for {number}.pdf in the folder
you give it. Real PDFs are copied to data/pdfs/. Every circular gets one row in
data/pdf_manifest.csv saying what happened to it:
    ok        real PDF with text, ready to ingest
    no_text   real PDF but scanned (pictures of pages), needs OCR later
    not_pdf   the "PDF" was really an error message (e.g. "404 NOT FOUND")
    damaged   real PDF that pypdf can't read
    missing   no {number}.pdf in the folder (never downloaded, or saved as .html)
Safe to run again: PDFs already in data/pdfs/ are re-checked but not copied again.
"""
# Docstring above = documentation for the whole file (what it does + how to run it).

import csv  # standard library: read circulars.csv, write pdf_manifest.csv
import logging  # standard library: controls which warning messages libraries print
import shutil  # standard library: "shell utilities", here shutil.copy2() to copy files
import sys  # standard library: sys.argv = the words you typed after the script name
from collections import Counter  # standard library: counts things, e.g. how many "ok"
from pathlib import Path  # standard library: file paths as objects, e.g. Path("data") / "pdfs"

from pypdf import PdfReader  # third-party (`uv add pypdf`): opens PDFs and extracts their text
from pypdf.errors import PdfReadError  # the error pypdf raises when a PDF is damaged

# Constants (ALL_CAPS by convention = values that don't change while the script runs).
CATALOG = Path("data/circulars.csv")  # input: one row per circular
PDF_DIR = Path("data/pdfs")  # output: the real PDFs (our raw / bronze layer)
MANIFEST = Path("data/pdf_manifest.csv")  # output: the checklist, one row per circular

# pypdf prints "fontTools is required..." warnings for some fonts. They don't affect the
# text we extract, so only show messages at ERROR level or worse.
logging.getLogger("pypdf").setLevel(logging.ERROR)


# "-> dict" = returns a dictionary like {"status": "ok", "pages": 3, "chars": 5120}.
def check_pdf(path: Path) -> dict:
    """Look inside one file and say what it really is."""
    # Check 1, "magic bytes": every real PDF starts with the 5 bytes %PDF-.
    # BSP's "404 NOT FOUND" pages were saved as .pdf; the name lies, the bytes don't.
    # read_bytes() reads the file as raw bytes; [:5] = the first 5. b"..." = a bytes literal.
    if path.read_bytes()[:5] != b"%PDF-":
        return {"status": "not_pdf", "pages": 0, "chars": 0}

    # Check 2: can pypdf open it and pull text out of every page?
    try:  # "try" = run this, and if it raises an error, jump to "except" instead of crashing
        pages = PdfReader(path).pages  # list of page objects
        # extract_text() can return None for an empty page; "or ''" turns that into "".
        # "".join(...) glues all the pages' text into one long string.
        text = "".join(page.extract_text() or "" for page in pages)
    except PdfReadError:  # catch only "damaged PDF" errors, not every possible bug
        return {"status": "damaged", "pages": 0, "chars": 0}

    # Check 3: a scanned PDF is just pictures of pages, so it has (almost) no text.
    # strip() removes spaces/newlines so a page of blank space doesn't count as text.
    chars = len(text.strip())
    status = "ok" if chars >= 100 else "no_text"  # one-line if/else ("conditional expression")
    return {"status": status, "pages": len(pages), "chars": chars}


def main(download_dir: str) -> None:
    # expanduser() turns "~/Downloads" into "/Users/<you>/Downloads".
    source = Path(download_dir).expanduser()
    # Create data/pdfs/ if needed. parents=True also creates data/; exist_ok=True = no error
    # if it's already there (so the script can run again).
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    # Read the catalog. DictReader turns each CSV line into a dict keyed by the header,
    # e.g. {"circular_no": "1242", "issued_on": "2026-11-10", ...}. We only need the numbers.
    with CATALOG.open(encoding="utf-8") as f:
        numbers = [row["circular_no"] for row in csv.DictReader(f)]

    manifest = []  # one dict per circular; becomes pdf_manifest.csv
    for number in numbers:
        target = PDF_DIR / f"{number}.pdf"  # where it should end up: data/pdfs/1242.pdf
        found = source / f"{number}.pdf"  # where Safari saved it: ~/Downloads/1242.pdf

        # Idempotent: if we already have it, check our copy instead of copying again.
        if target.exists():
            result = check_pdf(target)
        elif found.exists():
            result = check_pdf(found)
            # Keep every REAL PDF, even scanned ones (OCR can use them later).
            # Error messages pretending to be PDFs are not data, so they stay behind.
            if result["status"] != "not_pdf":
                # copy2 copies the file AND its metadata (e.g. the modified date).
                # Copy, not move: if anything goes wrong, the original is still in Downloads.
                shutil.copy2(found, target)
        else:  # never downloaded, or saved as 1242.html (File Not Found)
            result = {"status": "missing", "pages": 0, "chars": 0}

        # {"circular_no": ..., **result} = a new dict with the number first, then
        # everything from result ("**" unpacks a dict into another dict).
        manifest.append({"circular_no": number, **result})

    # Write the checklist. Same pattern as write_csv in make_catalog.
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["circular_no", "status", "pages", "chars"])
        writer.writeheader()
        writer.writerows(manifest)

    # Summary: Counter counts each status, e.g. Counter({"ok": 121, "missing": 207, ...}).
    counts = Counter(row["status"] for row in manifest)
    print(f"{len(manifest)} circulars -> {MANIFEST}")
    for status, count in counts.most_common():  # most_common() = biggest count first
        print(f"  {status:8} {count}")  # :8 pads the word to 8 characters so numbers line up


# Only runs when started directly (uv run python -m ingest.collect_pdfs ...), not on import.
if __name__ == "__main__":
    # sys.argv[1:] = everything after the script name, e.g. ["/Users/<you>/Downloads"];
    # * unpacks the list into arguments: main("/Users/<you>/Downloads")
    main(*sys.argv[1:])
