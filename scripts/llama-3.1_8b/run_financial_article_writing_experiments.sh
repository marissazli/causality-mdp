ARGS="$@"
python3 run_experiments.py --environment financial_article_writing --adversarial-agent CHIEF_EDITOR --model-client llama3.1:8b $ARGS
python3 run_experiments.py --environment financial_article_writing --adversarial-agent RESEARCHER --model-client llama3.1:8b $ARGS
python3 run_experiments.py --environment financial_article_writing --adversarial-agent ASSISTANT --model-client llama3.1:8b $ARGS
python3 run_experiments.py --environment financial_article_writing --adversarial-agent EDITOR --model-client llama3.1:8b $ARGS
python3 run_experiments.py --environment financial_article_writing --adversarial-agent IMAGE_GENERATOR  --model-client llama3.1:8b $ARGS
python3 run_experiments.py --environment financial_article_writing --adversarial-agent WRITER --model-client llama3.1:8b $ARGS