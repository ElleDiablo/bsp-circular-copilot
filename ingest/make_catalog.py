# Turn BSP's excel export into a clean catalog + a page of download links
# Usage: uv run python -m ingest.make_catalog data/Regulations.xlsx

# writes:
#   - data/circulars.csv        one row per circular
#   - data/download_links.html  one link per circular

import csv  # standard library: read/write CSV files
import html  # standard library: html.escape() makes text safe to put in HTML
import re  # standard library: regular expressions (patterns like "1232(corrected copy)")
import sys  # standard library: sys.argv and sys.exit
from pathlib import Path  # standard library: file paths as objects

from openpyxl import load_workbook  # third-party (`uv add openpyxl`): reads Excel files

PDF_URL = "https://www.bsp.gov.ph/Regulations/Issuances/{year}/{prefix}{number}.pdf" # template for PDF link
NUMBER_RE = re.compile(r"^\s*(\d+)\s*(?:\((.*)\))?\s*$")
    # Pattern that splits "1232(corrected copy)" into "1232" and "corrected copy".
    #   ^              start of the text
    #   \s*            any spaces (zero or more)
    #   (\d+)          group 1: one or more digits  -> the circular number
    #   \s*            any spaces (e.g. "1128 (Corrected Copy)" has one)
    #   (?:\((.*)\))?  optional "( ... )"; group 2 is whatever is inside the brackets -> the note
    #   \s*$           any spaces, then the end of the text
    # Plain numbers like "1242" also match; group 2 is then None.
    # re.compile() builds the pattern once so it isn't rebuilt for every row.
FIRST_NEW_STYLE = 1108 # from 1108 its just the number:2026/1242.pdf

# Take file path, return list of dictionaries
def read_rows(xlsx_path:str) -> list[dict]:
    sheet = load_workbook(xlsx_path, read_only=True).worksheets[0]
    rows = list(sheet.iter_rows(values_only=True))
    # rows[0] is title "Regulations", rows[1] header, rows[2] data
    if rows[1] != ("Number", "Date Issued", "Subject"):
        sys.exit(f"Unexpected header: {rows[1]} (did BSP change the export?)")

    catalog = []
    for number, issued_on, title in rows[2:]:
        match = NUMBER_RE.match(str(number))
        if not match:
            sys.exit(f"Can't read circular number: {number!r}")
        
        circular_no, note = match.groups()
        catalog.append({
            "circular_no": circular_no,
            "issued_on": issued_on,
            "title": title.strip(),
            "note": (note or "").strip().lower(),
            "source_url": PDF_URL.format(year=issued_on[:4], prefix="c" if int(circular_no) < FIRST_NEW_STYLE else "", number=circular_no),
        })
    return catalog

# writes file, returns nothing
def write_csv(catalog: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:     # auto closes "with", "w" - write, newline stops csv module addling blank lines in windows, utf-8 so letters like ñ survive
        writer = csv.DictWriter(f, fieldnames=list(catalog[0])) # DictWriter writes dics as rows, keys of 1st row become the column names
        writer.writeheader()        # header, first line, circular_no, issued_on, title, note, source_url
        writer.writerows(catalog)   # one line per circular document

def write_links(catalog: list[dict], path: Path) -> None:
    # build one <li> list item per circular, then join them with newlines into one string
    # f"..." pieces written next to each other are glued into single string
    items = "\n".join(
        f'<li><a href="{c["source_url"]}">{c["circular_no"]}</a> '  # clickable number
        f'({c["issued_on"]}) {html.escape(c["title"])}</li>'        # escape
        for c in catalog      # for every circular do the line above
    )

    # writing a tiny web page: heading, instructions, then numbered list (<ol>).
    path.write_text(
        f"<!doctype html><meta charset='utf-8'><title>BSP circulars</title>"
        f"<h1>{len(catalog)} BSP circulars</h1><p>Option-click each link to download.</p>"
        f"<ol>\n{items}\n</ol>\n",
        encoding="utf-8",        
    )

def main(xlsx_path: str, out_dir: str = "data") -> None:
    # step 1: read + clean
    catalog = read_rows(xlsx_path)

    # Data quality check: circular_no is the primary key in the database, so it must be unique.
    numbers = [c["circular_no"] for c in catalog]
    if len(numbers) != len(set(numbers)):
        sys.exit("Duplicate circular numbers in the export")

    out = Path(out_dir)
    write_csv(catalog, out / "circulars.csv")
    write_links(catalog, out / "download_links.html")

    noted = sum(1 for c in catalog if c["note"])
    print(f"{len(catalog)} circulars ({noted} with a note) -> {out}/circulars.csv, download_links.html")


# This block only runs when you start the file directly (python -m ingest.make_catalog ...),
# not when another file imports it. That lets tests import read_rows() without writing files.
if __name__ == "__main__":
    # sys.argv = ["…/make_catalog.py", "data/Regulations.xlsx"]; [1:] skips the script name,
    # and * passes the rest as arguments: main("data/Regulations.xlsx")
    main(*sys.argv[1:])
