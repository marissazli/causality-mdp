ARGS="$@"
python3 run_experiments.py --environment travel_planning --adversarial-agent PLANNER_AGENT --model-client llama3.1:8b $ARGS
python3 run_experiments.py --environment travel_planning --adversarial-agent RECOMMENDER_AGENT --model-client llama3.1:8b $ARGS
python3 run_experiments.py --environment travel_planning --adversarial-agent WEATHER_AGENT --model-client llama3.1:8b $ARGS
python3 run_experiments.py --environment travel_planning --adversarial-agent MESSAGING_AGENT --model-client llama3.1:8b $ARGS
python3 run_experiments.py --environment travel_planning --adversarial-agent TICKETING_AGENT --model-client llama3.1:8b $ARGS