.PHONY: all preprocess featurize ngrams train fast test

all: preprocess featurize ngrams train

preprocess:
	python pipeline.py preprocess

featurize:
	python pipeline.py featurize

ngrams:
	python pipeline.py ngrams

train:
	python pipeline.py train

fast:
	python pipeline.py all --fast 2000

test:
	pytest -q
