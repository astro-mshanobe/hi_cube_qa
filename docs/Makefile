# Build the hi_cube_qa documentation (reference + quickstart).
#   make            -> build both PDFs
#   make clean      -> remove LaTeX build artifacts (keeps the PDFs)
#   make distclean  -> also remove the PDFs

DOCS = hi_cube_qa hi_cube_qa_quickstart
PDFS = $(DOCS:=.pdf)
LATEX = pdflatex -interaction=nonstopmode -halt-on-error
AUX = *.aux *.log *.out *.toc *.bbl *.blg

all: $(PDFS)

# Full LaTeX -> BibTeX -> LaTeX x2 cycle for resolved references.
%.pdf: %.tex refs.bib hcqa-style.sty
	$(LATEX) $*
	-bibtex $*
	$(LATEX) $*
	$(LATEX) $*

clean:
	rm -f $(AUX)

distclean: clean
	rm -f $(PDFS)

.PHONY: all clean distclean
