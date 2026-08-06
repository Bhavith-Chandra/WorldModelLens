SPHINXBUILD ?= python -m sphinx
SOURCEDIR   = docs
BUILDDIR    = docs/_build

.PHONY: docs docs-clean docs-linkcheck

docs:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

docs-clean:
	$(SPHINXBUILD) -M clean $(SOURCEDIR) $(BUILDDIR)

docs-linkcheck:
	$(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck
