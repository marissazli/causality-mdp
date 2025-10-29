ARGS="$@"
python3 run_experiments.py --environment multi_agent_debate --adversarial-agent agent_0 $ARGS
python3 run_experiments.py --environment multi_agent_debate --adversarial-agent agent_1 $ARGS
python3 run_experiments.py --environment multi_agent_debate --adversarial-agent agent_2 $ARGS
python3 run_experiments.py --environment multi_agent_debate --adversarial-agent agent_3 $ARGS
python3 run_experiments.py --environment multi_agent_debate --adversarial-agent agent_4 $ARGS