ARGS="$@"
python3 run_experiments.py --environment financial_article_writing --adversarial-agent CHIEF_EDITOR  --model-client gpt-4.1 $ARGS
python3 run_experiments.py --environment financial_article_writing --adversarial-agent RESEARCHER  --model-client gpt-4.1 $ARGS
python3 run_experiments.py --environment financial_article_writing --adversarial-agent ASSISTANT  --model-client gpt-4.1 $ARGS
python3 run_experiments.py --environment financial_article_writing --adversarial-agent EDITOR  --model-client gpt-4.1 $ARGS
python3 run_experiments.py --environment financial_article_writing --adversarial-agent IMAGE_GENERATOR  --model-client gpt-4.1 $ARGS
python3 run_experiments.py --environment financial_article_writing --adversarial-agent WRITER  --model-client gpt-4.1 $ARGS