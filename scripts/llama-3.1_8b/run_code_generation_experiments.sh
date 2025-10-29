ARGS="$@"
python3 run_experiments.py --environment code_generation --adversarial-agent PROGRAMMER --model-client llama3.1:8b $ARGS
python3 run_experiments.py --environment code_generation --adversarial-agent CEO --model-client llama3.1:8b $ARGS
python3 run_experiments.py --environment code_generation --adversarial-agent DESIGNER --model-client llama3.1:8b $ARGS
python3 run_experiments.py --environment code_generation --adversarial-agent BROWSER --model-client llama3.1:8b $ARGS
python3 run_experiments.py --environment code_generation --adversarial-agent REVIEWER --model-client llama3.1:8b $ARGS
python3 run_experiments.py --environment code_generation --adversarial-agent TESTER --model-client llama3.1:8b $ARGS
python3 run_experiments.py --environment code_generation --adversarial-agent EXECUTER --model-client llama3.1:8b $ARGS