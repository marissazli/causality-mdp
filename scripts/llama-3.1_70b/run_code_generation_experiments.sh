ARGS="$@"
python3 run_experiments.py --environment code_generation --adversarial-agent PROGRAMMER $ARGS
python3 run_experiments.py --environment code_generation --adversarial-agent CEO $ARGS
python3 run_experiments.py --environment code_generation --adversarial-agent DESIGNER $ARGS
python3 run_experiments.py --environment code_generation --adversarial-agent BROWSER $ARGS
python3 run_experiments.py --environment code_generation --adversarial-agent REVIEWER $ARGS
python3 run_experiments.py --environment code_generation --adversarial-agent TESTER $ARGS
python3 run_experiments.py --environment code_generation --adversarial-agent EXECUTER $ARGS