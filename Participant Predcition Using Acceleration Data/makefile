SHELL := /usr/bin/env bash

venv/bin/activate: requirements.txt
	python3 -m venv venv ; \
	(\
	source ./venv/bin/activate ; \
	pip install -r requirements.txt ; \
	python -m ipykernel install --user --name venv ; \
	)
	
