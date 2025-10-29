ARGS="$@"
python3 run_experiments.py --environment travel_planning --adversarial-agent PLANNER_AGENT $ARGS
python3 run_experiments.py --environment travel_planning --adversarial-agent RECOMMENDER_AGENT $ARGS
python3 run_experiments.py --environment travel_planning --adversarial-agent WEATHER_AGENT $ARGS
python3 run_experiments.py --environment travel_planning --adversarial-agent MESSAGING_AGENT $ARGS
python3 run_experiments.py --environment travel_planning --adversarial-agent TICKETING_AGENT $ARGS