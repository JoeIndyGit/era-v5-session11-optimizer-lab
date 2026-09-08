.PHONY: all test clean
all:
	python run_all.py

test:
	pytest -q

clean:
	rm -f artifacts/*.csv artifacts/*.png artifacts/*.json
