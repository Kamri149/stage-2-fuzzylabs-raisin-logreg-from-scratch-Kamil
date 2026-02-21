.PHONY: install train serve docker-build docker-run

install:
	pip install -r requirements.txt

train:
	python train.py

serve:
	python server.py

docker-build:
	docker build -t raisin-infer .

docker-run:
	docker run -p 8000:8000 raisin-infer