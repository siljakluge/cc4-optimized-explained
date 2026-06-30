# CC4 Overleaf Document

This folder is a standalone LaTeX/Overleaf project for the optimized CAGE Challenge 4 documentation.

Entry point:

```bash
main.tex
```

Recommended Overleaf upload:

- `main.tex`
- `references.bib`
- `sections/*.tex`

Local build, if LaTeX is installed:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The document is intentionally marked as a living draft. Before treating it as final, refresh benchmarks from one evaluation script and mark each simulation audit item as fixed, open, or documented design choice.
