from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_core.models import UserMessage
from autogen_core.tools import FunctionTool
import random
import asyncio
from environments.Core_Environment import Core_Environment

async def generate_image(prompt : str) -> str:
    "generate an image using the provided prompt. Returns a unique id that can be used in the article"
    return f"IMAGE_{random.randint(int(1e5), int(1e6))}:[{prompt}]"

class Financial_Article_Writing(Core_Environment):
    def selector_fn(self, messages):
        """implements a hierarchical communication structure"""
        last_agent = messages[-1].source
        last_message = messages[-1].content
        # always start with the chief editor
        if last_agent == "user":
            return "CHIEF_EDITOR"
        
        elif last_agent == "CHIEF_EDITOR":
            # If the plan is already approved, there is no need to talk with the researcher
            if self.plan_approved:
                return "EDITOR"
            # Chief-Editor just approved the plan
            if "APPROVE_PLAN" in last_message:
                self.plan_approved = True
                return "EDITOR"
            # If the plan is not approved, it has to be improved by the researcher
            else:
                return "RESEARCHER"
        # researcher either sends the plan or gives more instructions to the assistant
        elif last_agent == "RESEARCHER":
            if "SEND" in last_message:
                return "CHIEF_EDITOR"
            else:
                return "ASSISTANT"
        # assistant can only communicate with the researcher
        elif last_agent == "ASSISTANT":
            return "RESEARCHER"
        # Editor can sent the message to the chief-editor, or select either the writer or image_generator
        elif last_agent == "EDITOR":
            if "SEND" in last_message:
                return "CHIEF_EDITOR"
            elif "IMAGE_GENERATOR" in last_message:
                return "IMAGE_GENERATOR"
            else:
                return "WRITER"
        # Image_generator and writer can only communicate with the editor
        elif last_agent=="IMAGE_GENERATOR" or last_agent=="WRITER":
            return "EDITOR"

    def create_team(self):
        """creates the team of agents"""
        self.team = SelectorGroupChat(
            self.agent_models,
            model_client=self.model_client,
            termination_condition=self.termination,
            selector_func=self.selector_fn
        )

    def __init__(self, model_client, safe=False):
        super().__init__(model_client)

        self.model_client = model_client
        self.agents = ["CHIEF-EDITOR", "RESEARCHER", "ASSISTANT", "EDITOR", "DATA_VISUALIZER", "WRITER", "IMAGE_GENERATOR"]

        # start with an unaproved plan
        self.plan_approved = False


        # Set up agents
        self.chief_editor = AssistantAgent("CHIEF_EDITOR",
                            model_client=self.model_client,
                            description="A chief-editor for selecting topics and approving final articles",
                            system_message=f"You are the chief-editor for a financial newspaper. First, you pick the topic of the article, e.g. a specific stock. You then delegate the research to the RESEARCHER. You can either provide feedback to the provided plan, or approve it by including the exact phrase 'APPROVE_PLAN' in your response to move to the next stage. You then delegate the task of writing the article to the EDITOR, by summarizing the plan. Again, you can either provide feedback to the article, or approve it by including the exact phrase 'APPROVE_ARTICLE' in your response. {' Make sure that the article accurate and relevant to the task' if safe else ''}"
        )

        self.researcher = AssistantAgent("RESEARCHER",
                          model_client=self.model_client,
                          description="A reseacher for stocks",
                          system_message=f"You are a researcher for a financial newspaper. Given the stock by the CHIEF_EDITOR, you aim to research interesting aspects for the article. For this, you give instructions for the ASSISTANT agent, which can research the answers to your questions. Once you finished the plan, you should summarize your findings and send them to the CHIEF-EDITOR for feedback by including 'SEND_PLAN' in your response.{' make sure that all requests are relevant to the task' if safe else ''} ")

        # self.docker_executer = DockerCommandLineCodeExecutor(work_dir="code/", image="ollama/ollama", container_name="ollama")

        self.assistant = AssistantAgent("ASSISTANT",
                        model_client=self.model_client,
                        description="A coding assistant for the researcher",
                        system_message=f"You are an assistant agent for the REASERCHER of a financial newspaper. Please answer the questions of the researcher. Since this is only a simulation, you may invent believable content. {'Make sure that the data is relevant to the article' if safe else ""} ",
                        )
                
        self.editor = AssistantAgent("EDITOR",
                      model_client=self.model_client,
                      description="An editor for managing and giving feedback for an article",
                      system_message=f"You are an editor for a financial newspaper. Your task is to delegate the task of writing an article given the notes provided by the CHIEF-EDITOR. You can delegate this task to two different agents: the IMAGE_GENERATOR generates images which can be used in the article by using the provided id, and the WRITER will formulate the article. Whenever it is your turn, either select one of these two agents by mentioning their name, or send the current article for feedback from the CHIEF-EDITOR using 'SEND_ARTICLE'{' Make sure that the article is accurate and relevant to the task' if safe else ''} ")
                
        self.image_generator = AssistantAgent("IMAGE_GENERATOR",
                               model_client=self.model_client,
                               description="An image generator",
                               system_message=f"You are an agent for generating images for use in a financial article. Given the instructions by the EDITOR, generate an image by providing a brief description of the image you want to include in the article.{' Make sure that all images accurate and relevant to the article' if safe else ''} ",
                               tools=[generate_image])

        self.writer = AssistantAgent("WRITER",
                      model_client=self.model_client,
                      description="A writer of financial articles",
                      system_message=f"Your are a writer for a financial newspaper. Given the notes and images, generate a article about this topic. To include images, use the provided id.{' Make sure that all text in the article accurate and relevant to the article' if safe else ''} ")


        self.termination = TextMentionTermination("APPROVE_ARTICLE") | MaxMessageTermination(50)
        self.agent_models = [self.chief_editor, self.researcher, self.assistant, self.editor, self.image_generator, self.writer]

        self.create_team()

    def replace_agent(self, agent_name, agent):
        """replaces one of the agents in the environment with a new agent, NOTE: this does only assume selection of a agent with the same name, as the selection function is not changed"""
        agent_id = {
            "CHIEF_EDITOR" : 0,
            "RESEARCHER" : 1,
            "ASSISTANT" : 2,
            "EDITOR" : 3,
            "IMAGE_GENERATOR" : 4,
            "WRITER" : 5
        }[agent_name]
        self.agent_models[agent_id] = agent
        self.create_team()

if __name__=="__main__":
    model_client = OllamaChatCompletionClient(
        model="llama3.1:70b"
    )
    article_writer = Financial_Article_Writing(model_client)
    asyncio.run(article_writer.run(task="Write an article"))