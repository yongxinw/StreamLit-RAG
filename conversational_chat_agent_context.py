import json
import os
import re
from typing import Any, List, Tuple

from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, ConversationalChatAgent, create_react_agent
from langchain.agents.conversational_chat.prompt import TEMPLATE_TOOL_RESPONSE
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain_community.llms import Tongyi
from langchain_core.agents import AgentAction
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


os.environ["DASHSCOPE_API_KEY"] = "sk-c92ed98926194b84a41a73db62af31d5"
CONTEXT_PATTERN = re.compile(r"^CONTEXT:")


class ConversationalChatAgentContext(ConversationalChatAgent):
    """
    An agent designed to hold a conversation in addition to using tools.
    This agent can ask for context from the user. To ask for context, tools have to return a prefix 'CONTEXT:' followed by the context question.
    """

    @property
    def _agent_type(self) -> str:
        raise NotImplementedError

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> List[BaseMessage]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts: List[BaseMessage] = []
        print(intermediate_steps)
        for action, observation in intermediate_steps:
            thoughts.append(AIMessage(content=action.log))
            if re.match(CONTEXT_PATTERN, observation):
                # remove the context_prefix from the observation
                # This is required to avoid the 'TEMPLATE_TOOL_RESPONSE' format response.
                human_message = HumanMessage(
                    content=re.sub(CONTEXT_PATTERN, "", observation)
                )
            else:
                human_message = HumanMessage(
                    content=TEMPLATE_TOOL_RESPONSE.format(observation=observation)
                )
            thoughts.append(human_message)
        return thoughts


class CheckUserCreditTool(BaseTool):
    """根据用户回答，检查用户学时状态"""

    name: str = "检查用户学时状态工具"
    description: str = (
        "用于检查用户学时状态，需要指通过 json 指定用户身份证号 user_id_number、用户想要查询的年份 year "
    )
    # args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    def _run(self, params) -> Any:

        params = params.replace("'", '"')
        print(params, type(params))
        try:
            params_dict = json.loads(params)
        except json.JSONDecodeError as e:
            print(e)
            return "请指定您的身份证号和想要查询学时的年份"
        CONTEXT_PROMPT = "You must ask the human about {context}. Reply with schema #2."

        if "user_id_number" not in params_dict:
            return CONTEXT_PROMPT.format(context="身份证号")
        if "year" not in params_dict:
            return CONTEXT_PROMPT.format(context="要查询的课程年份")
        # print(params)
        # try:
        #     params_dict = json.loads(params)
        # except json.JSONDecodeError:
        #     return "请指定您的身份证号和想要查询学时的年份"
        # if "user_id_number" in params_dict:
        #     user_id_number = params_dict["user_id_number"]

        # return result
        return "您的学时状态正常"


if __name__ == "__main__":
    credit_problem_prompt = PromptTemplate.from_template(
        """Answer the user's question step by step. Don't give the whole answer at once. Guide the user to the solution.

First check the user's learning method belongs to 电脑浏览器 or 手机微信扫码

If the user's learning method belongs to 电脑浏览器 or 手机微信扫码, then say 电脑浏览器请不要使用IE、edge等自带浏览器，可以使用搜狗、谷歌、360浏览器极速模式等浏览器试试。
Otherwise, say 目前支持的学习方式是电脑浏览器或者手机微信扫码两种，建议您再使用正确的方式试试

If the user's still has problems, then say 建议清除浏览器或者微信缓存再试试

If the user used the right method and 清除了缓存, then say，抱歉，您的问题涉及到测试，建议您联系平台的人工热线客服或者在线客服进行反馈

{chat_history}
Question: {input}
"""
    )
    credit_problem_prompt.input_variables = [
        # "agent_scratchpad",
        "input",
        "chat_history",
        # "tool_names",
        # "tools",
    ]
    chat_llm = Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm_chain = LLMChain(
        llm=chat_llm,
        memory=memory,
        prompt=credit_problem_prompt,
        verbose=True,
        output_key="output",
    )

    import ipdb

    ipdb.set_trace()

    tools = [CheckUserCreditTool()]
    qa_agent = ConversationalChatAgent.from_llm_and_tools(
        chat_llm,
        tools,
        # credit_problem_prompt
        memory=memory,
        system_message=credit_problem_prompt.template,
    )
    agent = AgentExecutor.from_agent_and_tools(
        agent=qa_agent, tools=tools, verbose=True, memory=memory
    )
    # agent.run(input)
    import ipdb

    ipdb.set_trace()
