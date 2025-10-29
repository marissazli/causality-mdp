ARGS="$@"
python3 run_experiments.py --environment financial_article_writing --adversarial-agent CHIEF_EDITOR $ARGS
python3 run_experiments.py --environment financial_article_writing --adversarial-agent RESEARCHER $ARGS
python3 run_experiments.py --environment financial_article_writing --adversarial-agent ASSISTANT $ARGS
python3 run_experiments.py --environment financial_article_writing --adversarial-agent EDITOR $ARGS
python3 run_experiments.py --environment financial_article_writing --adversarial-agent IMAGE_GENERATOR $ARGS
python3 run_experiments.py --environment financial_article_writing --adversarial-agent WRITER $ARGS