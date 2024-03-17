import json
import os
import sys
from typing import Any, List, Optional, Type
import streamlit as st
from langchain.retrievers import ContextualCompressionRetriever

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
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

# from langchain_core.tools import BaseTool
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.llms import Tongyi
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda

from statics import REGISTRATION_STATUS

os.environ["DASHSCOPE_API_KEY"] = "sk-91ee79b5f5cd4838a3f1747b4ff0e850"

st.set_page_config(
    page_title="大众云学智能客服平台",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("大众云学智能客服平台, powered by LangChain")
if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "欢迎您来到大众云学，我是大众云学的专家助手，我可以回答关于大众云学的所有问题。",
        }
    ]

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
            ret_str = "  \n".join(ret_str)
            return "经查询，您在大众云学平台上的注册状态如下：  \n" + ret_str
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
            return "您好，目前我们支持的用户类型为专技个人，用人单位，主管部门和继续教育机构，请确认您的用户类型。"
        if "user_role" not in params_dict:
            return "您好，目前我们支持的用户类型为专技个人，用人单位，主管部门和继续教育机构，请确认您的用户类型。"
        user_role = params_dict["user_role"]
        if user_role not in ["专技个人", "用人单位", "主管部门", "继续教育机构"]:
            return "您好，目前我们支持的用户类型为专技个人，用人单位，主管部门和继续教育机构，请确认您的用户类型。"
        agent_executor.agent.runnable.get_prompts()[0].template = (

"""Your ONLY job is to use a tool to answer the following question.

You MUST use a tool to answer the question. 
Simply Answer "抱歉，根据我的搜索结果，我无法回答这个问题" if you don't know the answer.
DO NOT answer the question without using a tool.

Current user role is """ + user_role + """.

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
        )
        return f"更新您的用户角色为{user_role}, 请问有什么可以帮到您？"


class AskForUserRoleTool(BaseTool):
    """询问用户角色"""

    name: str = "用户角色询问工具"
    description: str = "用于询问用户的角色，无需输入参数"
    # args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    # def _run(self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None) -> Any:
    def _run(self, params) -> Any:
        return "请问您是专技个人、用人单位、主管部门，还是继续教育机构？请先确认您的用户类型，以便我能为您提供相应的信息。"

@st.cache_data
def create_retrieval_tool(
    markdown_path,
    tool_name,
    tool_description,
    chunk_size: int = 100,
    chunk_overlap: int = 30,
    separators: List[str] = None,
    search_kwargs: dict = None,
    return_retriever: bool = False,
    rerank: bool = False,
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

    # if rerank:
    #     # compressor = CohereRerank()
    #     compressor = FlashrankRerank()
    #     retriever = ContextualCompressionRetriever(
    #         base_compressor=compressor, base_retriever=retriever
    #     )

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
    "./policies/registration/registration.md",
    "registration_engine",
    "查询大众云学平台注册流程，来回答如何注册的相关问题，并返回结果",
)

auditing_tool = create_retrieval_tool(
    "./policies/registration/auditing.md",
    "auditing_engine",
    "回答关于注册审核的相关问题，返回最相关的文档，如：账号怎么在审核？如何查询审核状态？",
)

withdrawal_tool = create_retrieval_tool(
    "./policies/registration/withdrawal_and_modification.md",
    "withdrawal_engine",
    "回答关于如何撤回、修改、驳回注册的问题，返回最相关的文档，如：注册后如何撤回？专技个人能撤回吗？怎么撤回不了？",
)

faq_personal_tool = create_retrieval_tool(
    "./policies/registration/professional_individual_reg_page_faq.md",
    "professional_individual_registration_faq_engine",
    "回答专技个人注册页面细项，以及常见问题，返回最相关的文档，如：证件号提示已存在，没有自己专业，单位找不到，没有单位怎么办，职称系列怎么选择",
)

# faq_employing_unit_tool = create_retrieval_tool(
#     "./policies/registration/employing_unit_reg_page_faq.md",
#     "employing_unit_registration_faq_engine",
#     "回答用人单位注册页面细项，以及常见问题，返回最相关的文档，如：单位性质和级别怎么选，单位所属行业选什么，主管部门怎么选/什么意思、上级单位是什么意思/怎么选，同级人社选什么，信息选错了怎么办",
# )

# faq_cont_edu_tool = create_retrieval_tool(
#     "./policies/registration/continuing_edu_inst_reg_page_faq.md",
#     "continuing_education_institute_registration_faq_engine",
#     "回答继续教育机构注册页面细项，以及常见问题，返回最相关的文档，如：机构级别怎么选、什么意思，行业主管部门是什么意思、怎么选，同级人社部门怎么选/同级人社呢，选错了怎么办/选的不对会有什么影响",
# )

# cannot_register_tool = create_retrieval_tool(
#     "./policies/registration/cannot_register.md",
#     "cannot_register_engine",
#     "回答用户无法注册的相关问题，返回最相关的文档，如：注册不了怎么办，注册不上怎么办，注册不了，无法注册，注册保存以后什么反应也没有，注册没反应",
#     search_kwargs={"k": 1},
# )

# login_problems_tool = create_retrieval_tool(
#     "./policies/registration/login_problems.md",
#     "login_problems_engine",
#     "回答用户登录问题的相关问题，返回最相关的文档，如：登录不了、无法登录、怎么登录不上",
#     search_kwargs={"k": 1},
#     chunk_size=100,
#     separators=["\n\n"],
# )

# login_problems_detail_tool = create_retrieval_tool(
#     "./policies/registration/login_problems_details.md",
#     "login_problems_detail_engine",
#     "回答用户登录问题的细节相关问题，返回最相关的文档，如：没有滑块，找不到滑块，登录为什么提示验证失败，哪里有滑块，密码错误，忘记密码，账号不存在，登录显示审核中",
#     search_kwargs={"k": 1},
#     chunk_size=100,
#     separators=["\n\n"],
# )

# TODO: Add more here
# forgot_password_tool = create_retrieval_tool(
#     "./policies/registration/forgot_password.md",
#     "forgot_password_engine",
#     "回答用户忘记密码的相关问题，返回最相关的文档，如：忘记密码怎么办，密码忘记了，找回密码",
#     # search_kwargs={"k": 1},
#     # chunk_size=100,
#     # separators=["\n\n"]
# )

# create operation retrievers
individual_operation_tool = create_retrieval_tool(
    "./policies/operation/individual_operation.md",
    "individual_operation_engine",
    "回答专技个人学时、学时申报、修改单位的系统操作相关问题，返回最相关的文档，如：怎么学时申报，如何提交学时，为什么不能学时申报，学时申报信息天填错了怎么办，学时信息填好后无法保存，我怎么不能学时申报、我的账号里怎么没有学时申报，证书和发明专利能申报、抵扣多少学时。再如：怎么修改单位，修改单位的话，现在单位能知道吗，现在单位审核吗，单位调转提示有待审核信息，不能修改单位，单位调转信息填错怎么办，怎么删除人员，离职的人员怎么办，怎么调到临时单位",
    # search_kwargs={"k": 5},
    chunk_size=100,
    separators=["\n\n"],
)

employing_unit_operation_tool = create_retrieval_tool(
    "./policies/operation/employing_unit_operation.md",
    "employing_unit_operation_engine",
    "回答用人单位学时申报、注册审核、信息变更、更换管理员、人员信息查询的系统操作相关问题，返回最相关的文档，如：单位怎么审核，怎么把人员调出单位，人员离职了怎么调出去，如何审核人员提交的学时，学时申报错了，单位也审核了怎么办，怎么驳回，学时申报错了，单位也审核了怎么办，单位培训计划，怎么提交、审核，怎么更换单位超级管理员，单位如何增加管理员，如何查询单位名下专技人员信息",
    chunk_size=100,
    separators=["\n\n"],
)

supervisory_department_operation_tool = create_retrieval_tool(
    "./policies/operation/supervisory_department_operation.md",
    "supervisory_department_operation_engine",
    "回答主管部门注册审核、信息变更、继续教育机构审核、单位调转审核、学时申报审核、人员信息查询的系统操作相关问题，返回最相关的文档，如：如何审核单位或个人注册信息、人员或用人单位信息变更审核、如何审核继教机构信息、人员调入和单位调转审核操作、如何审核专技人员的学时、学时报错了，怎么驳回、学时申报错了，也审核通过了，还能驳回吗、如何查询主管部门下面单位情况",
    chunk_size=100,
    separators=["\n\n"],
)

# create modify info retrievers
personal_modify_info_tool = create_retrieval_tool(
    "./policies/modify_info/professional_person_modify_info.md",
    "professional_person_modify_info_engine",
    "回答专技个人注册信息修改相关问题，返回最相关的文档，如：怎么改姓名、姓名错了能改吗、怎么改身份证号码，怎么修改单位/单位错了怎么改、怎么修改单位区域、单位区域错了怎么改，怎么修改手机号、手机号错了，怎么修改，怎么修改职称、职称换了，怎么改",
    search_kwargs={"k": 5},
    chunk_size=200,
    separators=["\n\n"],
)

employ_supervise_modify_info_tool = create_retrieval_tool(
    "./policies/modify_info/employ_supervise_modify_info.md",
    "employ_supervise_modify_info_engine",
    "回答用人单位、主管部门注册信息修改相关问题，返回最相关的文档，如：怎么更换超级管理员、管理员能更换吗，修改用人单位/主管部门账号手机号、邮箱，怎么修改单位名称，怎么修改统一信用代码，如何查询上级部门、上级部门管理员信息，怎么修改单位区域、注册地/单位地址，怎么更换单位上级部门",
    search_kwargs={"k": 5},
    chunk_size=200,
    separators=["\n\n"],
)

cont_edu_modify_info_tool = create_retrieval_tool(
    "./policies/modify_info/cont_edu_modify_info.md",
    "cont_edu_modify_info_engine",
    "回答继续教育机构的注册信息修改相关问题，返回最相关的文档，如：怎么更换超级管理员、管理员能更换吗，修改继续教育机构账号手机号、邮箱，怎么修改单位名称，怎么修改统一信用代码，如何查询上级部门、上级部门管理员信息，怎么修改单位区域、注册地/单位地址，怎么更换单位上级部门",
    search_kwargs={"k": 7},
    chunk_size=400,
    separators=["\n\n"],
)

# complaints
complaints_tool = create_retrieval_tool(
    "./policies/complaints/complaints.md",
    "complaints_engine",
    "回答用户投诉相关问题，返回最相关的文档，如：建议增加人员删除功能，建议单位账号可以不使用管理员身份证号，可以自己设置，浮动公告飘的太快、遮挡信息，关闭按钮不明显，不方便关闭，客服联系方式遮挡信息，建议设置关闭按钮，查询统计的数据、怎么导出数据、怎么导出单位所有人的学习情况的数据，退休人员的账号怎么办、退休人员怎么调出本单位、怎么删除退休人员的账号，为什么不能手机网页登陆、手机网页不能登录、页面显示不全，建议添加新的专业的课程、课程里没有我的专业，有没有课件、没有课件讲解吗，课程不能倍速播放、视频播放太慢了，购买怎么不能一起支付、课程怎么一块买",
    search_kwargs={"k": 10},
    chunk_size=100,
    separators=["\n\n"],
)

# policy inquiry
policy_inquiry_tool = create_retrieval_tool(
    "./policies/policy_inquiry/policy_inquiry.md",
    "policy_inquiry_engine",
    "回答用户政策咨询相关问题，返回最相关的文档，如：职称评审什么时候、职称评审有什么要求，新一年继续教育学习时间、什么时候能报名学习、往年的课程还能补学吗、报名时间，报职称有什么要求，我需要继续教育吗、每年都需要继续教育吗，为什么要继续教育",
    search_kwargs={"k": 5},
    chunk_size=100,
    separators=["\n\n"],
)

# other questions
other_questions_tool = create_retrieval_tool(
    "./policies/other_questions/other_questions.md",
    "other_questions_engine",
    "回答用户其他问题，返回最相关的文档，如：会计人员需要几年继续教育、会计人员在哪里学习、会计人员需要学习公需课吗、会计的怎么补学、卫生技术在哪里学习、医护人员在哪里学习、卫生技术专业怎么补学， 平台上怎么收费，省直单位公需课怎么收费、课程没学完怎么办、怎么开发票，有卫健委的电话吗、有人社电话吗、有主管部门电话吗、人社电话是哪一个、职称评审部门电话是什么，评职称需要什么条件，评职称需要学习几年继续教育，怎么和贵平台合作、想和你们合作，怎么联系，买课收费吗、学习要交费吗、为什么要收费、能便宜吗、有优惠吗，怎么注销账号、我要把账号注销",
    search_kwargs={"k": 8},
    chunk_size=100,
    separators=["\n\n"],
)


# Create Agent
model = Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3})
# model.model_name = "qwen-max"
# model.model_kwargs = {"temperature": 0.3}

tools = [
    # multiply,
    RegistrationStatusTool(),
    AskForUserRoleTool(),
    UpdateUserRoleTool(),
    registration_tool,
    auditing_tool,
    withdrawal_tool,
    faq_personal_tool,
    # faq_employing_unit_tool,
    # faq_cont_edu_tool,
    # cannot_register_tool,
    # login_problems_tool,
    # login_problems_detail_tool,
    # forgot_password_tool,
    individual_operation_tool,
    employing_unit_operation_tool,
    supervisory_department_operation_tool,
]

# DO NOT hallucinate!!! You MUST use a tool to collect information to answer the questions!!! ALWAYS use a tool to answer a question if possible. Otherwise, you MUST ask the user for more information.
prompt = hub.pull("hwchase17/react")
prompt.template = """Your ONLY job is to use a tool to answer the following question.

You MUST use a tool to answer the question. 
Simply Answer "抱歉，根据我的搜索结果，我无法回答这个问题" if you don't know the answer.
DO NOT answer the question without using a tool.

Current user role is unknown.

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

memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True
)


# create a router chain
class CheckUserRoleTool(BaseTool):
    """根据用户回答，检查用户角色"""

    name: str = "检查用户角色工具"
    description: str = "用于检查用户在对话中的角色，无需输入参数 "
    # args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    def _run(self, params) -> Any:
        print(params)
        template = agent_executor.agent.runnable.get_prompts()[0].template.lower()
        # print(template)
        start_index = template.find("current user role is") + len(
            "current user role is"
        )
        end_index = template.find("\n", start_index)
        result = template[start_index:end_index].strip()
        return result


router_prompt = hub.pull("hwchase17/react")
router_prompt.template = """Your ONLY job is to determine the user role. DO NOT Answer the question.

You MUST use a tool to find out the user role.
DO NOT hallucinate!!!!

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you will not answer
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!


Question: {input}
Thought:{agent_scratchpad}
user role:
"""
router_prompt.input_variables = [
    "agent_scratchpad",
    "input",
    "tool_names",
    "tools",
]

router_tools = [CheckUserRoleTool()]

router_chain = create_react_agent(
    Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
    router_tools,
    router_prompt,
)
router_chain_executor = AgentExecutor.from_agent_and_tools(
    agent=router_chain, tools=router_tools, verbose=True, handle_parsing_errors=True
)

user_role_prompt = hub.pull("hwchase17/react")
user_role_prompt.template = """Your ONLY job is to ask the user to provide their role information regardless of the input.

You MUST ALWAYS say: 请问您是专技个人、用人单位、主管部门，还是继续教育机构？请先确认您的用户类型，以便我能为您提供相应的信息。
You MUST use a tool to update user role.
DO NOT hallucinate!!!!

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


Question: {input}
Thought:{agent_scratchpad}
"""
user_role_prompt.input_variables = [
    "agent_scratchpad",
    "input",
    "tool_names",
    "tools",
]

user_role_tools = [UpdateUserRoleTool()]

user_role_chain = create_react_agent(
    Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
    user_role_tools,
    user_role_prompt,
)
user_role_chain_executor = AgentExecutor.from_agent_and_tools(
    agent=user_role_chain, tools=user_role_tools, verbose=True, handle_parsing_errors=True
)


# general_chain = (
#     PromptTemplate.from_template(
#         """Respond to the following question:

# Question: {input}
# Answer:"""
#     )
#     | Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3})
# )


def route(info):
    print(info["topic"])
    if "unknown" in info["topic"]["output"].lower():
        return user_role_chain_executor
    return agent_executor

full_chain = {"topic": router_chain_executor, "input": lambda x: x["input"]} | RunnableLambda(route)

# update prompt with this: agent_executor.agent.runnable.get_prompts()[0]


if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    # st.session_state.chat_engine = index.as_chat_engine(
    #     chat_mode="condense_question", verbose=True
    # )
    st.session_state.chat_engine = full_chain

if prompt := st.chat_input(
    "您的问题"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # st.session_state.chat_engine.memory.add_message(
    #     {"role": "user", "content": prompt}
    # )

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # response = st.session_state.chat_engine.chat(prompt)
            # response = st.session_state.chat_engine.invoke({"input": prompt})
            response = st.session_state.chat_engine.invoke({"input": prompt})
            st.write(response["output"])
            message = {"role": "assistant", "content": response["output"]}
            # st.session_state.chat_engine.memory.add_message(message)
            st.session_state.messages.append(message)  # Add response to message history
