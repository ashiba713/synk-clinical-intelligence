.PHONY: setup data train train-smoke evaluate predict api dashboard tests docker-audit

setup:
	pip install -r requirements.txt
	pip install -e . --no-deps

data:
	python -m scripts.cli generate-data

train:
	python -m scripts.cli train

train-smoke:
	python -m scripts.cli train --set training.max_epochs=2 --set synthetic.n_patients=40

evaluate:
	python -m scripts.cli evaluate --model-dir artifacts/models/synk_both/v1 --ablation

predict:
	python -m scripts.cli predict --encounter SYN-E-00001 --explain

api:
	python -m scripts.cli serve

dashboard:
	python -m scripts.cli dashboard

tests:
	pytest tests/ -q

mlflow:
	mlflow server --host 127.0.0.1 --port 5000

docker-audit:
	python -m scripts.cli train --skip-training
