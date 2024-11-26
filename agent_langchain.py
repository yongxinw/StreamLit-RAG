import json
import os
import sys
from typing import Any, List, Optional, Type

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.embeddings.dashscope import DashScopeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.pydantic_v1 import BaseModel, Field
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

# from langchain_core.tools import BaseTool
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.llms import Tongyi
from langchain_community.vectorstores import FAISS

from statics import REGISTRATION_STATUS

os.environ["DASHSCOPE_API_KEY"] = "sk-91ee79b5f5cd4838a3f1747b4ff0e850"


# Simple demo tool - a simple calculator
class SimpleCalculatorTool(BaseTool):
    """计算两个输的乘积的简单计算器"""

    name: str = "简单计算器"
    description: str = (
        "用于计算两个数的乘积，需要指通过 json 指定第一个数 first_number、第二个数 second_number "
    )
    # args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    # def _run(self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None) -> Any:
    def _run(self, params) -> Any:
        print(params)
        # print(type(params))
        params_dict = json.loads(params)
        return params_dict["first_number"] * params_dict["second_number"]


class RegistrationStatusTool(BaseTool):
    """查询用户在大众云学平台上的注册状态"""

    name: str = "注册状态查询工具"
    description: str = (
        "用于查询用户在大众云学平台上的注册状态，需要指通过 json 指定用户身份证号 user_id_number "
    )
    # args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    # def _run(self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None) -> Any:
    def _run(self, params) -> Any:
        print(params)
        # print(type(params))
        try:
            params_dict = json.loads(params)
        except json.JSONDecodeError:
            return "请指定您或者管理员身份证号"
        if "user_id_number" not in params_dict:
            return "请指定您或者管理员身份证号"
        try:
            int(params_dict["user_id_number"])
        except ValueError:
            return "请指定您或者管理员身份证号"
        input = params_dict["user_id_number"]
        if REGISTRATION_STATUS.get(input) is not None:
            status = REGISTRATION_STATUS.get(input)
            ret_str = [f"{k}: {v}" for k, v in status.items()]
            ret_str = "\n".join(ret_str)
            return "经查询，您在大众云学平台上的注册状态如下：\n" + ret_str
        return "经查询，您尚未在大众云学平台上注册"


class UpdateUserRoleTool(BaseTool):
    """根据用户回答，更新用户角色"""

    name: str = "用户角色更新工具"
    description: str = (
        "用于更新用户在对话中的角色，需要指通过 json 指定用户角色 user_role "
    )
    # args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    # def _run(self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None) -> Any:
    def _run(self, params) -> Any:
        print(params)
        try:
            params_dict = json.loads(params)
        except json.JSONDecodeError:
            return "请指定正确的用户角色"
        if "user_role" not in params_dict:
            return "请指定正确的用户角色"
        user_role = params_dict["user_role"]
        if user_role not in ["专技个人", "用人单位", "主管部门", "继续教育机构"]:
            return "您好，目前我们支持的用户类型为专技个人，用人单位，主管部门和继续教育机构，请确认您的用户类型。 "
        agent_executor.agent.runnable.get_prompts()[0].template = (
            """Answer the following questions as best as you can.

Current user role is"""
            + user_role
            + """
If user role is unknown, ALWAYS ask for user role before doing ANYTHING.
DO NOT take any actions without knowing user role.
Possible user roles are: 专技个人，用人单位，主管部门，继续教育机构

WHENEVER the user has provided user role, use the correct tool to update user role.
After updating user role, You MUST continue answering the original questions.

If the user is unsure of whether they have registered, you MUST ask them to provide the 管理员身份证号 or 上级单位统一信用代码 and THEN use the right tool to check the registration status. 
If the user cannot contact 管理员 or 上级审核部门, you MUST ask them to provide the 管理员身份证号 or 上级单位统一信用代码 and THEN use the right tool to check the registration status. 

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

{chat_history}
Question: {input}
Thought:{agent_scratchpad}
"""
        )
        # print(agent_executor.memory.chat_memory.messages)
        # original_question = agent_executor.memory.chat_memory.messages[-2].content

        return "更新您的用户角色为" + user_role
        # return agent_executor.invoke({"input": original_question})["output"]
        # print(type(params))


class AskForUserRoleTool(BaseTool):
    """询问用户角色"""

    name: str = "用户角色询问工具"
    description: str = "用于询问用户的角色，无需输入参数"
    # args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    # def _run(self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None) -> Any:
    def _run(self, params) -> Any:
        return "请问您是专技个人、用人单位、主管部门，还是继续教育机构？请先确认您的用户类型，以便我能为您提供相应的信息。"


def create_retrieval_tool(
    markdown_path,
    tool_name,
    tool_description,
    chunk_size: int = 100,
    chunk_overlap: int = 30,
    separators: List[str] = None,
    search_kwargs: dict = None,
    return_retriever: bool = False,
):
    # Load files
    loader = UnstructuredMarkdownLoader(markdown_path)
    docs = loader.load()
    print(docs)

    # Declare the embedding model
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
    )

    # Chunk the files
    # chunk_size = 100
    # chunk_overlap = 30

    if separators is not None:
        text_splitter = RecursiveCharacterTextSplitter(
            # text_splitter = CharacterTextSplitter(
            separators=["\n\n"],
            # separators=["\n\n", "\n"], is_separator_regex=True
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    # if "login_problems" in markdown_path:
    #     import ipdb

    #     ipdb.set_trace()
    # Create a retriever tool
    if search_kwargs is None:
        retriever = vector.as_retriever()
    else:
        retriever = vector.as_retriever(search_kwargs=search_kwargs)

    # retriever

    registration_tool = create_retriever_tool(
        retriever,
        name=tool_name,
        description=tool_description,
    )
    # import ipdb
    # ipdb.set_trace()
    if return_retriever:
        return registration_tool, retriever
    return registration_tool


# create registration retrievers
registration_tool = create_retrieval_tool(
    "/Users/yongxinw/Developer/llamaindex-chat-with-streamlit-docs/policies/registration/registration.md",
    "registration_engine",
    "查询大众云学平台注册流程，来回答如何注册的相关问题，并返回结果",
)

auditing_tool = create_retrieval_tool(
    "/Users/yongxinw/Developer/llamaindex-chat-with-streamlit-docs/policies/registration/auditing.md",
    "auditing_engine",
    "回答关于注册审核的相关问题，返回最相关的文档，如：账号怎么在审核？如何查询审核状态？",
)

withdrawal_tool = create_retrieval_tool(
    "/Users/yongxinw/Developer/llamaindex-chat-with-streamlit-docs/policies/registration/withdrawal_and_modification.md",
    "withdrawal_engine",
    "回答关于如何撤回、修改、驳回注册的问题，返回最相关的文档，如：注册后如何撤回？专技个人能撤回吗？怎么撤回不了？",
)

faq_personal_tool = create_retrieval_tool(
    "/Users/yongxinw/Developer/llamaindex-chat-with-streamlit-docs/policies/registration/professional_individual_reg_page_faq.md",
    "professional_individual_registration_faq_engine",
    "回答专技个人注册页面细项，以及常见问题，返回最相关的文档，如：证件号提示已存在，没有自己专业，单位找不到，没有单位怎么办，职称系列怎么选择",
)

faq_employing_unit_tool = create_retrieval_tool(
    "/Users/yongxinw/Developer/llamaindex-chat-with-streamlit-docs/policies/registration/employing_unit_reg_page_faq.md",
    "employing_unit_registration_faq_engine",
    "回答用人单位注册页面细项，以及常见问题，返回最相关的文档，如：单位性质和级别怎么选，单位所属行业选什么，主管部门怎么选/什么意思、上级单位是什么意思/怎么选，同级人社选什么，信息选错了怎么办",
)

faq_cont_edu_tool = create_retrieval_tool(
    "/Users/yongxinw/Developer/llamaindex-chat-with-streamlit-docs/policies/registration/continuing_edu_inst_reg_page_faq.md",
    "continuing_education_institute_registration_faq_engine",
    "回答继续教育机构注册页面细项，以及常见问题，返回最相关的文档，如：机构级别怎么选、什么意思，行业主管部门是什么意思、怎么选，同级人社部门怎么选/同级人社呢，选错了怎么办/选的不对会有什么影响",
)

cannot_register_tool = create_retrieval_tool(
    "/Users/yongxinw/Developer/llamaindex-chat-with-streamlit-docs/policies/registration/cannot_register.md",
    "cannot_register_engine",
    "回答用户无法注册的相关问题，返回最相关的文档，如：注册不了怎么办，注册不上怎么办，注册不了，无法注册，注册保存以后什么反应也没有，注册没反应",
    search_kwargs={"k": 1},
)

login_problems_tool = create_retrieval_tool(
    "/Users/yongxinw/Developer/llamaindex-chat-with-streamlit-docs/policies/registration/login_problems.md",
    "login_problems_engine",
    "回答用户登录问题的相关问题，返回最相关的文档，如：登录不了、无法登录、怎么登录不上",
    search_kwargs={"k": 1},
    chunk_size=100,
    separators=["\n\n"],
)

login_problems_detail_tool = create_retrieval_tool(
    "/Users/yongxinw/Developer/llamaindex-chat-with-streamlit-docs/policies/registration/login_problems_details.md",
    "login_problems_detail_engine",
    "回答用户登录问题的细节相关问题，返回最相关的文档，如：没有滑块，找不到滑块，登录为什么提示验证失败，哪里有滑块，密码错误，忘记密码，账号不存在，登录显示审核中",
    search_kwargs={"k": 1},
    chunk_size=100,
    separators=["\n\n"],
)

forgot_password_tool = create_retrieval_tool(
    "/Users/yongxinw/Developer/llamaindex-chat-with-streamlit-docs/policies/registration/forgot_password.md",
    "forgot_password_engine",
    "回答用户忘记密码的相关问题，返回最相关的文档，如：忘记密码怎么办，密码忘记了，找回密码",
    # search_kwargs={"k": 1},
    # chunk_size=100,
    # separators=["\n\n"]
)

# create operation retrievers
study_hour_tool = create_retrieval_tool(
    "/Users/yongxinw/Developer/llamaindex-chat-with-streamlit-docs/policies/operation/study_hour_report.md",
    "study_hour_engine",
    "回答学时、学时申报相关问题，返回最相关的文档，如：怎么学时申报，如何提交学时，为什么不能学时申报，学时申报如何审核、驳回，学时申报错了怎么办，用人单位提交培训计划如何审核、怎么申报学时",
    # search_kwargs={"k": 1},
    chunk_size=100,
    separators=["\n\n"],
)

# Create Agent
model = Tongyi()
model.model_name = "qwen-max"
model.model_kwargs = {"temperature": 0.3}

tools = [
    # multiply,
    RegistrationStatusTool(),
    AskForUserRoleTool(),
    UpdateUserRoleTool(),
    registration_tool,
    auditing_tool,
    withdrawal_tool,
    faq_personal_tool,
    faq_employing_unit_tool,
    faq_cont_edu_tool,
    cannot_register_tool,
    login_problems_tool,
    login_problems_detail_tool,
    forgot_password_tool,
    study_hour_tool,
]

"When user role is unknown, you MUST use the registration_engine to fetch available roles and ask the user for his role based on registration_engine output"
"""
    WHENEVER WHENEVER the user has provided 用户角色, use the correct tool to update 用户角色 and proceed with the answering questions.
    Possible user roles are: 专技个人，用人单位，主管部门，继续教育机构

    If the user is unsure of whether they have registered, you MUST ask them to provide the 管理员身份证号 or 上级单位统一信用代码 and THEN use the right tool to check the registration status. 
    If the user cannot contact 管理员 or 上级审核部门, you MUST ask them to provide the 管理员身份证号 or 上级单位统一信用代码 and THEN use the right tool to check the registration status. 
    If the user cannot log in, you MUST first ask them what the error message is and THEN use the right tool to check the login problems."""

# DO NOT hallucinate!!! You MUST use a tool to collect information to answer the questions!!! ALWAYS use a tool to answer a question if possible. Otherwise, you MUST ask the user for more information.
prompt = hub.pull("hwchase17/react")
prompt.template = """Answer the following questions as best as you can. DO NOT take any actions without knowing user role. 
ALWAYS base your answers on outputs from tools. DO NOT hallucinate!!!!
Keep your answers short. Do not provide more information than necessary.

Current user role is unknown
If user role is unknown, ALWAYS ask for user role before doing ANYTHING.

Possible user roles are: 专技个人，用人单位，主管部门，继续教育机构

WHENEVER the user has provided user role, use the correct tool to update user role.
After updating user role, You MUST continue answering the original questions.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

{chat_history}
Question: {input}
Thought:{agent_scratchpad}
"""
prompt.input_variables = [
    "agent_scratchpad",
    "input",
    "tool_names",
    "tools",
    "chat_history",
]

memory = ConversationBufferMemory(memory_key="chat_history")
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True
)

# update prompt with this: agent_executor.agent.runnable.get_prompts()[0]

import ipdb

ipdb.set_trace()

### BELOW IS FROM LLAMAINDEX
# agent_template_str = (
#     "\nYou are designed to help with a variety of tasks, from answering questions     to providing summaries to other types of analyses. \n\n"
#     "## Tools\nYou have access to a wide variety of tools. You are responsible for using\n"
#     "the tools in any sequence you deem appropriate to complete the task at hand.\n"
#     "This may require breaking the task into subtasks and using different tools\n"
#     "to complete each subtask.\n\n"
#     "You have access to the following tools:\n{tool_desc}\n\n## Output Format\n"
#     "To answer the question, please use the following format.\n\n"
#     "```\nThought: I need to use a tool to help me answer the question.\n"
#     "Action: tool name (one of {tool_names}) if using a tool.\n"
#     'Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})\n'
#     "```\n\nPlease ALWAYS start with a Thought.\n\n"
#     "Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.\n\n"
#     "If this format is used, the user will respond in the following format:\n\n"
#     "```\nObservation: tool response\n```\n\n"
#     "You should keep repeating the above format until you have enough information\n"
#     "to answer the question without using any more tools. At that point, you MUST respond\n"
#     "in the one of the following two formats:\n\n```\n"
#     "Thought: I can answer without using any more tools.\n"
#     "Answer: [your answer here]\n```\n\n```\n"
#     "Thought: I cannot answer the question with the provided tools.\n"
#     "Answer: Sorry, I cannot answer your query.\n```\n\n"
#     "ALWAYS check user role from chat history before any actions.\n"
#     "When user role is unknown, you MUST use the registration_engine to fetch available roles and ask the user for his role based on registration_engine output\n"
#     "You MUST NOT use any tools to infer user role or ask directly.\n"
#     "When user has provided role information, use the correct tool to update user role and proceed with the answering questions.\n"
#     "Current user role is unknown\n\n"
#     "All conversation is in Chinese. Please use Chinese for all conversation.\n\n"
#     "## Current Conversation\n"
#     "Below is the current conversation consisting of interleaving human and assistant messages.\n\n"
# )


# import os

# from statics import REGISTRATION_STATUS

# os.environ["DASHSCOPE_API_KEY"] = "sk-91ee79b5f5cd4838a3f1747b4ff0e850"


# def multiply(a: int, b: int) -> int:
#     """计算两个数的乘积并返回结果"""
#     return a * b


# def update_user_role(input: str = "123"):
#     """
#     根据语境，更新用户角色以及对应的回答模板，以便更好地回答用户问题。
#     """
#     # TODO: This is a hacky fix. A better solution: https://stackoverflow.com/questions/11283961/partial-string-formatting
#     USER_ROLE = input
#     policy_engine_tmpl_str = (
#         "注意：回答问题前，请先从用户对话中尝试确定用户角色，若无法推测，则询问用户角色，不要推测用户角色，回答问题时请保持技术性和基于事实，不要产生幻觉。\n"
#         #  "注意：若用户角色为未知，请先询问用户角色，不要推测用户角色，回答问题时请保持技术性和基于事实，不要产生幻觉。\n"
#         "注意：用户角色为{user_role}\n"
#         "语境信息如下\n"
#         "---------------------\n"
#         "{context_str}\n"
#         "---------------------\n"
#         "请根据语境信息，不要使用先验知识，回答下面的问题。\n"
#         "Query: {query_str}\n"
#         "Answer: "
#     )
#     print(USER_ROLE)
#     template_str = policy_engine_tmpl_str.format(
#         context_str="{context_str}",
#         user_role=USER_ROLE if USER_ROLE is not None else "未知",
#         query_str="{query_str}",
#     )
#     print(template_str)
#     policy_engine_tmpl = PromptTemplate(template_str)
#     agent.agent_worker._get_tools("")[0]._query_engine.update_prompts(
#         {"response_synthesizer:text_qa_template": policy_engine_tmpl}
#     )

#     agent_template_str = (
#         "\nYou are designed to help with a variety of tasks, from answering questions     to providing summaries to other types of analyses. \n\n"
#         "## Tools\nYou have access to a wide variety of tools. You are responsible for using\n"
#         "the tools in any sequence you deem appropriate to complete the task at hand.\n"
#         "This may require breaking the task into subtasks and using different tools\n"
#         "to complete each subtask.\n\n"
#         "You have access to the following tools:\n{tool_desc}\n\n## Output Format\n"
#         "To answer the question, please use the following format.\n\n"
#         "```\nThought: I need to use a tool to help me answer the question.\n"
#         "Action: tool name (one of {tool_names}) if using a tool.\n"
#         'Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})\n'
#         "```\n\nPlease ALWAYS start with a Thought.\n\n"
#         "Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.\n\n"
#         "If this format is used, the user will respond in the following format:\n\n"
#         "```\nObservation: tool response\n```\n\n"
#         "You should keep repeating the above format until you have enough information\n"
#         "to answer the question without using any more tools. At that point, you MUST respond\n"
#         "in the one of the following two formats:\n\n```\n"
#         "Thought: I can answer without using any more tools.\n"
#         "Answer: [your answer here]\n```\n\n```\n"
#         "Thought: I cannot answer the question with the provided tools.\n"
#         "Answer: Sorry, I cannot answer your query.\n```\n\n"
#         "ALWAYS check user role from chat history before any actions.\n"
#         "When user role is unknown, you MUST use the registration_engine to fetch available roles and ask the user for his role based on registration_engine output\n"
#         "You MUST NOT use any tools to infer user role or ask directly.\n"
#         "When user has provided role information, use the correct tool to update user role and proceed with the answering questions.\n"
#         "Current user role is" + USER_ROLE + "\n\n"
#         "If the user is unsure of whether they have registered, you MUST ask them to provide the administrator ID number and THEN use the right tool to check the registration status.\n\n"
#         "## IMPORTANT: \n"
#         "All conversation is in Chinese. Please use Chinese for all conversation.\n\n"
#         "## Current Conversation\n"
#         "Below is the current conversation consisting of interleaving human and assistant messages.\n\n"
#     )
#     agent.update_prompts(
#         {"agent_worker:system_prompt": PromptTemplate(agent_template_str)}
#     )
#     return "user role updated"


# def lookup_by_id(input: str = "123"):
#     """
#     根据用户输入的ID号，查询用户在大众云学平台上的注册状态。
#     """
#     if REGISTRATION_STATUS.get(input) is not None:
#         return REGISTRATION_STATUS.get(input)
#     return "经查询，您尚未在大众云学平台上注册"


# # multiply_tool = FunctionTool.from_defaults(
# #     fn=multiply,
# #     tool_metadata=ToolMetadata(name="multiply", description="计算两个数的乘积并返回结果。"),
# # )

# # update_user_role_tool = FunctionTool.from_defaults(
# #     fn=update_user_role,
# #     tool_metadata=ToolMetadata(
# #         name="update_role", description="根据语境，更新用户角色以及对应的回答模板，以便更好地回答用户问题。"
# #     ),
# # )

# # lookup_by_id_tool = FunctionTool.from_defaults(
# #     fn=lookup_by_id,
# #     tool_metadata=ToolMetadata(
# #         name="lookup_by_id",
# #         description="根据用户输入的ID号，查询用户在大众云学平台上的注册状态。",
# #     ),
# # )


# def get_query_engine_tool(input_dir, input_files, tool_name, description):
#     policy_engine_tmpl_str = (
#         "注意：回答问题前，请先从用户对话中尝试确定用户角色，若无法推测，则询问用户角色，不要推测用户角色，回答问题时请保持技术性和基于事实，不要产生幻觉。\n"
#         # "注意：若用户角色为未知，请先询问用户角色，不要推测用户角色，回答问题时请保持技术性和基于事实，不要产生幻觉。\n"
#         "注意：用户角色为未知\n"
#         "语境信息如下\n"
#         "---------------------\n"
#         "{context_str}\n"
#         "---------------------\n"
#         "请根据语境信息，不要使用先验知识，回答下面的问题。\n"
#         "Query: {query_str}\n"
#         "Answer: "
#     )
#     policy_engine_refine_tmpl_str = (
#         "以下是原始查询：{query_str}\n"
#         "我们已经提供了一个现有的答案：{existing_answer}\n"
#         "我们有机会通过下面的一些更多上下文来改进现有的答案（仅在需要时）。\n"
#         "------------\n"
#         "{context_msg}"
#         "------------\n"
#         "根据新的上下文，改进原始答案以更好地回答查询。如果上下文没有用，返回原始答案。\n"
#         "Refined Answer:"
#     )
#     index = load_data(input_dir, input_files)
#     policy_engine = index.as_query_engine(
#         similarity_top_k=5,
#         verbose=True,
#     )
#     policy_engine_tmpl = PromptTemplate(policy_engine_tmpl_str)
#     policy_enging_refine_tmpl = PromptTemplate(policy_engine_refine_tmpl_str)
#     policy_engine.update_prompts(
#         {"response_synthesizer:text_qa_template": policy_engine_tmpl}
#     )
#     policy_engine.update_prompts(
#         {"response_synthesizer:refine_template": policy_enging_refine_tmpl}
#     )
#     policy_query_tool = QueryEngineTool(
#         query_engine=policy_engine,
#         metadata=ToolMetadata(
#             name=tool_name,
#             description=description,
#         ),
#     )
#     return policy_query_tool


# tools = [
#     # policy_query_tool,
#     # multiply_tool,
#     get_query_engine_tool(
#         input_dir=None,
#         input_files=["./policies/registration/registration.md"],
#         tool_name="registration_engine",
#         description="负责查询大众云学平台的注册方法，返回最相关的文档",
#     ),
#     get_query_engine_tool(
#         input_dir=None,
#         input_files=["./policies/registration/auditing.md"],
#         tool_name="auditing_engine",
#         description="负责回答关于注册审核的相关问题，返回最相关的文档",
#     ),
#     get_query_engine_tool(
#         input_dir=None,
#         input_files=["./policies/registration/withdrawal_and_modification.md"],
#         tool_name="withdrawal_engine",
#         description="负责回答关于如何撤回、修改、驳回注册的问题，返回最相关的文档",
#     ),
#     get_query_engine_tool(
#         input_dir=None,
#         input_files=["./policies/registration/professional_individual_reg_page_faq.md"],
#         tool_name="professional_individual_registration_faq_engine",
#         description="负责回答专技个人注册页面细项，以及常见问题，返回最相关的文档",
#     ),
#     get_query_engine_tool(
#         input_dir=None,
#         input_files=["./policies/registration/employing_unit_reg_page_faq.md"],
#         tool_name="employing_unit_registration_faq_engine",
#         description="负责回答用人单位注册页面细项，以及常见问题，返回最相关的文档",
#     ),
#     get_query_engine_tool(
#         input_dir=None,
#         input_files=["./policies/registration/supervisory_department_reg_page_faq.md"],
#         tool_name="supervisory_department_registration_faq_engine",
#         description="负责回答主管部门注册页面细项，以及常见问题，返回最相关的文档",
#     ),
#     get_query_engine_tool(
#         input_dir=None,
#         input_files=["./policies/registration/continuing_edu_inst_reg_page_faq.md"],
#         tool_name="continuing_education_institute_registration_faq_engine",
#         description="负责回答继续教育机构注册页面细项，以及常见问题，返回最相关的文档",
#     ),
#     update_user_role_tool,
#     lookup_by_id_tool,
# ]

# # agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)
# # agent.update_prompts({"agent_worker:system_prompt": PromptTemplate(agent_template_str)})

# import ipdb

# ipdb.set_trace()
