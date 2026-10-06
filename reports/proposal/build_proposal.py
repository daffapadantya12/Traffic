"""Render the Assignment 1 proposal: Markdown -> docx, html, pdf.

Run from the project root:
    python3 reports/proposal/make_proposal_figures.py      # figures first (optional, once)
    python3 reports/proposal/build_proposal.py
Uses RStudio's bundled pandoc and headless Google Chrome, like the methodology build.
Prints a word count of the body (everything before the reference list) so the
1000-word limit of the brief can be checked.
"""
import re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PANDOC = "/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64/pandoc"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MD = HERE / "Assignment1_Proposal.md"


def word_count(text: str) -> dict:
    body = text.split("---", 2)[2]
    main = body.split("# References")[0]
    genai = body.split("# Use of generative AI")[1] if "# Use of generative AI" in body else ""
    def wc(s):   # words as a reader would count them: markdown syntax tokens are not words
        s = re.sub(r"!\[([^\]]*)\]\([^)]*\)\{[^}]*\}", r"\1", s)      # image -> its caption text
        s = re.sub(r"^\|[-| ]+\|\s*$", "", s, flags=re.M)               # table rule lines
        s = re.sub(r"[|#*]+", " ", s)                                     # pipes, hashes, emphasis marks
        return len(re.findall(r"\S+", s))
    captions = re.findall(r"!\[([^\]]*)\]", main)
    return dict(body_incl_headings_tables_captions=wc(main), captions=sum(wc(c) for c in captions), genai_statement=wc(genai))


def main():
    text = MD.read_text()
    print("word count:", word_count(text))
    css = ROOT / "reports/_build/style.css"
    base = ["--from", "markdown+smart", "--resource-path", str(HERE)]
    subprocess.run([PANDOC, str(MD), "-o", str(HERE / "Assignment1_Proposal.docx")] + base, check=True, cwd=HERE)
    subprocess.run([PANDOC, str(MD), "-o", str(HERE / "Assignment1_Proposal.html"), "--standalone", "--embed-resources",
                    "--section-divs", "--css", str(css)] + base, check=True, cwd=HERE)
    html = HERE / "Assignment1_Proposal.html"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=5000",
                    f"--print-to-pdf={HERE / 'Assignment1_Proposal.pdf'}", f"file://{html}"],
                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in ["Assignment1_Proposal.docx", "Assignment1_Proposal.html", "Assignment1_Proposal.pdf"]:
        p = HERE / f
        print(f, f"{p.stat().st_size/1e6:.2f} MB" if p.exists() else "MISSING")


if __name__ == "__main__":
    main()
