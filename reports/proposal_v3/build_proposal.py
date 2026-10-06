"""Render the v3 Assignment 1 proposal (Perempatan Fatmawati twin): Markdown -> docx, html, pdf.

Run from the project root:  python3 reports/proposal_v3/build_proposal.py
Prints the word count of the body (from "1. Introduction" to just before "References", including headings,
table cells and figure captions) and of the generative-AI statement, so the 1000-word limit can be checked.
"""
from __future__ import annotations

import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PANDOC = "/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64/pandoc"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MD = os.path.join(HERE, "Assignment1_Proposal.md")


def words(s: str) -> int:
    s = re.sub(r"!\[([^\]]*)\]\([^)]*\)(\{[^}]*\})?", r"\1", s)       # image -> caption text
    s = re.sub(r"^\|[-:| ]+\|\s*$", "", s, flags=re.M)              # table rule lines
    s = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", s, flags=re.M)        # list markers are not words
    s = re.sub(r"\$\$.*?\$\$", " FORMULA ", s, flags=re.S)          # a display formula counts as one word
    s = re.sub(r"\$[^$]+\$", " x ", s)                               # inline math counts as one word
    s = re.sub(r"[*#|]+", " ", s)
    return len(re.findall(r"\S+", s))


def main():
    text = open(MD, encoding="utf-8").read()
    body = text.split("---", 2)[2]
    main_part = body.split("# References")[0]
    genai = body.split("# Use of generative AI")[1] if "# Use of generative AI" in body else ""
    caps = re.findall(r"!\[([^\]]*)\]", main_part)
    print(f"words: body {words(main_part)} (captions {sum(words(c) for c in caps)}), generative-AI statement {words(genai)}, "
          f"total {words(main_part) + words(genai)}")
    css = os.path.join(ROOT, "reports", "_build", "style.css")
    base = ["--from", "markdown+smart+tex_math_dollars", "--resource-path", HERE]
    subprocess.run([PANDOC, MD, "-o", os.path.join(HERE, "Assignment1_Proposal.docx")] + base, check=True, cwd=HERE)
    html = os.path.join(HERE, "Assignment1_Proposal.html")
    subprocess.run([PANDOC, MD, "-o", html, "--standalone", "--embed-resources", "--section-divs", "--mathml", "--css", css] + base,
                   check=True, cwd=HERE)
    pdf = os.path.join(HERE, "Assignment1_Proposal.pdf")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=8000",
                    f"--print-to-pdf={pdf}", f"file://{html}"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in ("Assignment1_Proposal.docx", "Assignment1_Proposal.pdf"):
        p = os.path.join(HERE, f)
        print(f, f"{os.path.getsize(p)/1e6:.2f} MB" if os.path.exists(p) else "MISSING")


if __name__ == "__main__":
    main()
