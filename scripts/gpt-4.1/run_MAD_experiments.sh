ARGS="$@"
python3 run_experiments.py --environment multi_agent_debate --adversarial-agent agent_0  --model-client gpt-4.1 $ARGS
python3 run_experiments.py --environment multi_agent_debate --adversarial-agent agent_1  --model-client gpt-4.1 $ARGS
python3 run_experiments.py --environment multi_agent_debate --adversarial-agent agent_2  --model-client gpt-4.1 $ARGS
python3 run_experiments.py --environment multi_agent_debate --adversarial-agent agent_3  --model-client gpt-4.1 $ARGS
python3 run_experiments.py --environment multi_agent_debate --adversarial-agent agent_4  --model-client gpt-4.1 $ARGS