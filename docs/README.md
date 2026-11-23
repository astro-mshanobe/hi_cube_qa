# hi_cube_qa — documentation

LaTeX documentation for the `hi_cube_qa` toolkit. Two documents share one style
file and one bibliography.

| File | What it is |
|------|------------|
| `hi_cube_qa.tex` | **Reference manual** — every metric with its full scientific justification (noise, radiometer equation, beam, spectra, checklist), plus config reference and a symbol glossary. |
| `hi_cube_qa_quickstart.tex` | **Quickstart** — the three-step workflow, what each report panel shows, and the three settings that matter. Read this first. |
| `hcqa-style.sty` | Shared preamble: palette, code-listing style, and the TikZ title-page logo (no external image needed). |
| `refs.bib` | Bibliography (CARACal, MeerKAT, WSClean, Briggs, TMS, and the software stack). |
| `Makefile` | Build helper. |
| `*.pdf` | Pre-built outputs (regenerate with `make`). |

## Building

Requires a TeX distribution with `pdflatex` and `bibtex` (e.g. TeX Live, MacTeX).

```sh
make            # build both PDFs (pdflatex -> bibtex -> pdflatex x2)
make clean      # remove LaTeX build artifacts, keep the PDFs
make distclean  # also remove the PDFs
```

To build a single document by hand:

```sh
pdflatex hi_cube_qa
bibtex   hi_cube_qa
pdflatex hi_cube_qa
pdflatex hi_cube_qa
```

## Notes

- Editing the science/usage text: change the `.tex` files. Shared look-and-feel
  (colours, logo, listing style) lives in `hcqa-style.sty`.
- Adding a reference: add the entry to `refs.bib` and `\citep{key}` it in the text.
- The boresight ("P20") method is flagged in the reference manual's appendix as
  needing its proper bibliographic citation — fill that in before any formal use.
