import asyncio
from autogen_agentchat.ui import Console

class Core_Environment:
    """Implements the core functionality of an environment"""
    def __init__(self, model_client):
        self.model_client = model_client

    def reset(self):
        """reset the agent team, which resets the context"""
        asyncio.run(self.team.reset())

    async def run(self, task):
        """run the environment on the given task"""
        return await Console(self.team.run_stream(task = task))

    def add_agent(self, agents, selector_fn, termination):
        """add a new agent to the environment. This also updates the termination criteria and selector function"""
        self.selector_fn = selector_fn
        self.agent_models.append(agents)
        self.termination = self.termination | termination
        self.create_team()