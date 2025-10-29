
ARGS="$@"
python3 run_experiments.py --environment code_generation --adversarial-agent CEO --model-client gpt-4.1 $ARGS
python3 run_experiments.py --environment code_generation --adversarial-agent PROGRAMMER --model-client gpt-4.1 $ARGS
python3 run_experiments.py --environment code_generation --adversarial-agent DESIGNER --model-client gpt-4.1 $ARGS
python3 run_experiments.py --environment code_generation --adversarial-agent BROWSER --model-client gpt-4.1 $ARGS
python3 run_experiments.py --environment code_generation --adversarial-agent REVIEWER --model-client gpt-4.1 $ARGS
python3 run_experiments.py --environment code_generation --adversarial-agent TESTER --model-client gpt-4.1 $ARGS
python3 run_experiments.py --environment code_generation --adversarial-agent EXECUTER --model-client gpt-4.1 $ARGS