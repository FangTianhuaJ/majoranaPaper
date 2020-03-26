BASE_DIR = ../tmp
CODE_DIR = $(BASE_DIR)/code
PAPER_DIR = $(BASE_DIR)/paper

plot:
	cd .. && python paper-figures.py
	git ls-files *.svg | sed s/svg// | xargs -I % inkscape %svg --export-pdf=%pdf -C

pdf: get-template
	latexmk -pdf phasemajoranas

get-template:
	@if [ ! -f SciPost.cls ]; then\
		curl -O https://scipost.org/static/submissions/latex/SciPost.cls;\
	fi
	@if [ ! -f SciPost_bibstyle.bst ]; then\
		curl -O https://scipost.org/static/submissions/latex/SciPost_bibstyle.bst;\
	fi
paper:
	-mkdir -p $(BASE_DIR)
	-mkdir -p $(PAPER_DIR)
	cp phasemajoranas.bib $(PAPER_DIR)
	cp phasemajoranas.tex $(PAPER_DIR)
	cp SciPost.cls $(PAPER_DIR)
	cp SciPost_bibstyle.bst $(PAPER_DIR)
	export DATE=`date +%Y-%m-%d`
	sed -i -e "s/\\\today/`date +%Y-%m-%d`/g" $(PAPER_DIR)/phasemajoranas.tex
	-mkdir -p $(PAPER_DIR)/figures
	cp -r figures/*pdf $(PAPER_DIR)/figures/

arxiv: paper
	$(eval TARGET_DIR := $(BASE_DIR)/arxiv)
	-mkdir -p $(TARGET_DIR)
	cp phasemajoranas.bbl $(TARGET_DIR)
	-mv $(PAPER_DIR)/* $(TARGET_DIR)
	rm $(TARGET_DIR)/phasemajoranas.bib
	cd $(TARGET_DIR) && tar -cvzf ../phasemajoranas_arxiv.tar.gz *

code:
	mkdir -p $(BASE_DIR)
	mkdir -p $(CODE_DIR)
	cp ../README.md $(CODE_DIR)/
	cp ../LICENSE.txt $(CODE_DIR)/
	cp ../environment.yml $(CODE_DIR)/
	jupytext ../paper-figures.py --to ipynb -o $(CODE_DIR)/paper-figures.ipynb
	jupytext ../ham-plane-wave.py --to ipynb -o $(CODE_DIR)/have-plane-wave-basis.ipynb
	cp ../phasemajoranas.py $(CODE_DIR)/
	mkdir $(CODE_DIR)/paper
	mkdir $(CODE_DIR)/paper/figures

data: code
	$(eval TARGET_DIR := $(BASE_DIR)/data_sub)
	-mkdir -p $(TARGET_DIR)
	-mkdir -p $(TARGET_DIR)/data
	cp -R ../data/* $(TARGET_DIR)/data
	-mv $(CODE_DIR)/* $(TARGET_DIR)
	cd $(TARGET_DIR) && tar -cvzf ../phasemajoranas.tar.gz *

clean:
	rm -rf $(BASE_DIR)
