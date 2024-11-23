"""
TODO:
    0. refactor code.
    1. Chain not able to get back to the starting point if seleting a role in the beginning (explore lang-graph). 
    2. If similarity score is too low (asking irrellavant questions), answer can't answer.
    3. Add causual chatbot.
    4. Future data organization.
    5. Improve latency.
    6. Test general API for more data.
    7. Add jining 学时获取不到的问题
    8. Augmentation
    9. eval set
"""

from langchain.tools.render import render_text_description
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from operator import itemgetter

import json
import os
import re
import sys
from typing import Any, List, Optional, Type

import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.chains import LLMChain
from langchain.embeddings.dashscope import DashScopeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.pydantic_v1 import BaseModel, Field
from langchain.retrievers import ContextualCompressionRetriever
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
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda

import apis
from statics import (
    COURSE_PURCHASES,
    CREDIT_HOURS,
    LOC_STR,
    REGISTRATION_STATUS,
    REGISTRATION_STATUS_NON_IDV,
)
from utils import (
    check_user_location,
    create_atomic_retriever_agent,
    create_atomic_retriever_agent_single_tool_qa_map,
    create_dummy_agent,
    create_react_agent_with_memory,
    create_single_function_call_agent,
    output_parser,
)
from tools import *

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def _init(api_key="sk-91ee79b5f5cd4838a3f1747b4ff0e850"):
    os.environ["DASHSCOPE_API_KEY"] = api_key
    st.set_page_config(
        page_title="大众云学智能客服平台",
        page_icon="",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    st.title("大众云学智能客服平台")
    if (
        "messages" not in st.session_state.keys()
    ):  # Initialize the chat messages history
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": """欢迎您来到大众云学，我是大众云学的专家助手，我可以回答关于大众云学的所有问题。测试请使用身份证号372323199509260348。测试公需课/专业课学时，请使用年份2019/2020。测试课程购买，退款等，请使用年份2023，课程名称新闻专业课培训班。测试模拟数据如下：\n\n
            专技个人注册状态 = {
            "372323199509260348": {
                "状态": "已注册",
                "注册时间": "2021-03-01",
                "注册地点": "济南市",
                "管理员": "王芳芳",
                "角色": "专技个人",
                "单位": "山东省济南市中心医院",
            },
        }
    
        用人单位注册状态 = {
            "山东省济南市中心医院": {
                "状态": "已注册",
                "注册时间": "2020-03-01",
                "注册地点": "济南市",
                "管理员": "王芳芳",
                "角色": "用人单位",
                "上级部门": "山东省医疗协会",
            }
        }
    
        学时记录 = {
            "372323199509260348": {
                "2019": {
                    "公需课": [
                        {"课程名称": "公需课1", "学时": 10, "进度": 100, "考核": "合格"},
                        {"课程名称": "公需课2", "学时": 10, "进度": 100, "考核": "合格"},
                        {"课程名称": "公需课3", "学时": 10, "进度": 100, "考核": "未完成"},
                        {"课程名称": "公需课4", "学时": 10, "进度": 85, "考核": "未完成"},
                    ],
                    "专业课": [
                        {"课程名称": "专业课1", "学时": 10, "进度": 100, "考核": "合格"},
                        {"课程名称": "专业课2", "学时": 10, "进度": 100, "考核": "合格"},
                        {"课程名称": "专业课3", "学时": 10, "进度": 100, "考核": "未完成"},
                        {"课程名称": "专业课4", "学时": 10, "进度": 85, "考核": "未完成"},
                    ],
                },
                "2020": {
                    "公需课": [
                        {"课程名称": "公需课5", "学时": 10, "进度": 100, "考核": "未完成"},
                        {"课程名称": "公需课6", "学时": 10, "进度": 12, "考核": "未完成"},
                    ],
                    "专业课": [
                        {"课程名称": "专业课5", "学时": 10, "进度": 85, "考核": "未完成"},
                    ],
                },
            }
        }
    
        课程购买记录 = {
            "372323199509260348": {
                "2023": {
                    "新闻专业课培训班": {
                        "课程名称": "新闻专业课培训班",
                        "课程类别": "专业课",
                        "学时": 10,
                        "进度": 90,
                        "考核": "未完成",
                        "购买时间": "2023-01-01",
                        "购买地点": "山东省济南市",
                        "培训机构": "山东省新闻学院",
                    },
                },
                "2024": {
                    "新闻专业课培训班": {
                        "课程名称": "新闻专业课培训班",
                        "课程类别": "专业课",
                        "学时": 10,
                        "进度": 0,
                        "考核": "未完成",
                        "购买时间": "2024-01-01",
                        "购买地点": "山东省济南市",
                        "培训机构": "山东省新闻学院",
                    },
                },
            }
        }
        """,
            }
        ]


_init()


class RegistrationStatusToolIndividual(BaseTool):
    """查询专技个人在大众云学平台上的注册状态"""

    name: str = "专技个人注册状态查询工具"
    description: str = (
        "用于查询专技个人在大众云学平台上的注册状态，只有当用户明确提及需要帮助查询时调用，需要指通过 json 指定用户身份证号 user_id_number "
    )
    return_direct: bool = True

    def _run(self, params) -> Any:
        print(params)
        params_dict = params
        if "user_id_number" not in params_dict:
            return "抱歉，我还没有成功识别您的身份证号码，请指定"
        try:
            int(params_dict["user_id_number"])
        except Exception:
            return "抱歉，我还没有成功识别您的身份证号码，请指定"

        return apis.get_registration_status(params_dict)


class RegistrationStatusToolUniversal(BaseTool):
    """查询用户在大众云学平台上的注册状态"""

    name: str = "统一注册状态查询工具"
    description: str = (
        "用于查询用户在大众云学平台上的注册状态，只有当用户明确提及需要帮助查询时调用，需要指通过 json 指定查询号码 user_id_number "
    )
    return_direct: bool = True

    def _run(self, params) -> Any:
        print(params)
        params_dict = params
        if "user_id_number" not in params_dict:
            return "抱歉，我还没有成功识别您的身份证号码，单位信用代码，或者单位名称，请指定"
        try:
            int(params_dict["user_id_number"])
        except Exception:
            try:
                str(params_dict["user_id_number"])
            except Exception:
                return "抱歉，我还没有成功识别您的身份证号码，单位信用代码，或者单位名称，请指定"
        input = str(params_dict["user_id_number"])
        if input in ["unknown", "未知"]:
            return "抱歉，我还没有成功识别您的身份证号码，单位信用代码，或者单位名称，请指定"

        return apis.get_registration_status(params_dict)


class RegistrationStatusToolNonIndividual(BaseTool):
    """查询用人单位、主管部门或继续教育机构在大众云学平台上的注册状态"""

    name: str = "非个人注册状态查询工具"
    description: str = (
        "用于查询用人单位、主管部门或继续教育机构在大众云学平台上的注册状态，只有当用户明确提及需要帮助查询时调用，需要指通过 json 指定用户身份证号 user_id_number "
    )
    return_direct: bool = True

    def _run(self, params) -> Any:
        print(params)
        try:
            params_dict = json.loads(params)
        except json.JSONDecodeError:
            return "抱歉，我还没有成功识别您的单位管理员身份证号或者单位名称或者统一信用代码，请指定"
        if "user_id_number" not in params_dict:
            return "抱歉，我还没有成功识别您的单位管理员身份证号或者单位名称或者统一信用代码，请指定"
        try:
            str(params_dict["user_id_number"])
        except ValueError:
            return "抱歉，我还没有成功识别您的单位管理员身份证号或者单位名称或者统一信用代码，请指定"
        input = str(params_dict["user_id_number"])

        return apis.get_registration_status(params_dict)


class UpdateUserRoleTool2(BaseTool):
    """根据用户回答，更新用户角色"""

    name: str = "用户角色更新工具"
    description: str = (
        "用于更新用户在对话中的角色，需要指通过 json 指定用户角色 user_role "
    )
    # args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    # def _run(self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None) -> Any:
    def _run(self, params) -> Any:
        print(params, type(params))
        if isinstance(params, str):
            try:
                params_dict = json.loads(params)
            except json.JSONDecodeError:
                return '您好，目前我们支持的用户类型为专技个人，用人单位，主管部门和继续教育机构，请问您想咨询那个用户类型？（回复"跳过"默认进入专技个人用户类型）'
        elif isinstance(params, dict):
            params_dict = params
        else:
            return '您好，目前我们支持的用户类型为专技个人，用人单位，主管部门和继续教育机构，请问您想咨询那个用户类型？（回复"跳过"默认进入专技个人用户类型）'
        if params_dict is None:
            return '您好，目前我们支持的用户类型为专技个人，用人单位，主管部门和继续教育机构，请问您想咨询那个用户类型？（回复"跳过"默认进入专技个人用户类型）'
        # try:
        #     params_dict = json.loads(params)
        # except json.JSONDecodeError:
        #     return "您好，目前我们支持的用户类型为专技个人，用人单位，主管部门和继续教育机构，请确认您的用户类型。"
        if "user_role" not in params_dict:
            return '您好，目前我们支持的用户类型为专技个人，用人单位，主管部门和继续教育机构，请问您想咨询那个用户类型？（回复"跳过"默认进入专技个人用户类型）'
        if params_dict["user_role"] is None:
            return '您好，抱歉我没有检测到您提供的用户类型，目前我们支持的用户类型为专技个人，用人单位，主管部门和继续教育机构，请问您想咨询那个用户类型？（回复"跳过"默认进入专技个人用户类型）'
        # if not isinstance(params_dict["user_role"], dict):
        #     return '您好，抱歉我没有检测到您提供的用户类型，目前我们支持的用户类型为专技个人，用人单位，主管部门和继续教育机构，请问您想咨询那个用户类型？（回复"跳过"默认进入专技个人用户类型）'

        # user_role = list(params_dict["user_role"].values())[0]
        user_role = params_dict["user_role"]
        if user_role not in [
            "专技个人",
            "用人单位",
            "主管部门",
            "继续教育机构",
            "跳过",
        ]:
            return '您好，目前我们支持的用户类型为专技个人，用人单位，主管部门和继续教育机构，请确认您的用户类型。（回复"跳过"默认进入专技个人用户类型）'
        if user_role == "跳过":
            user_role = "专技个人"
        main_qa_agent_executor.agent.runnable.get_prompts()[0].template = (
            """Your ONLY job is to use a tool to answer the following question.

You MUST use a tool to answer the question. 
Simply Answer "您能提供更多关于这个问题的细节吗？" if you don't know the answer.
DO NOT answer the question without using a tool.

Current user role is """
            + user_role
            + """.

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
        # st.session_state.user_role = user_role
        return f"更新您的用户角色为{user_role}, 请问有什么可以帮到您？"


@st.cache_resource
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
    use_cached_faiss: bool = True,
):
    # Load files
    loader = UnstructuredMarkdownLoader(markdown_path)
    docs = loader.load()

    # Declare the embedding model
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
    )
    # embeddings = OpenAIEmbeddings()

    if os.path.exists(f"./vector_store/{tool_name}.faiss") and use_cached_faiss:
        vector = FAISS.load_local(
            f"./vector_store/{tool_name}.faiss",
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        # =============only split by separaters============
        documents = []
        all_content = docs[0].page_content
        texts = [sentence for sentence in all_content.split("\n\n")]
        meta_data = [docs[0].metadata] * len(texts)
        for content, meta in zip(texts, meta_data):
            new_doc = Document(page_content=content, metadata=meta)
            documents.append(new_doc)

        print(documents)
        vector = FAISS.from_documents(documents, embeddings)

        vector.save_local(f"./vector_store/{tool_name}.faiss")
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

    registration_tool.return_direct = True
    if return_retriever:
        return registration_tool, retriever
    return registration_tool


# CREATE RETRIEVERS
individual_qa_tool = create_retrieval_tool(
    "./policies_v2/individual_q.md",
    "individual_qa_engine",
    "回答个人用户的相关问题，返回最相关的文档",
    search_kwargs={"k": 3},
    chunk_size=100,
    separators=["\n\n"],
    use_cached_faiss=True,
)

employing_unit_qa_tool = create_retrieval_tool(
    "./policies_v2/employing_unit_q.md",
    "employing_unit_qa_engine",
    "回答用人单位用户的相关问题，返回最相关的文档",
    search_kwargs={"k": 3},
    chunk_size=100,
    separators=["\n\n"],
    use_cached_faiss=True,
)

supervisory_department_qa_tool = create_retrieval_tool(
    "./policies_v2/supervisory_dept_q.md",
    "supervisory_department_qa_engine",
    "回答主管部门用户的相关问题，返回最相关的文档",
    search_kwargs={"k": 3},
    chunk_size=100,
    separators=["\n\n"],
    use_cached_faiss=True,
)

cont_edu_qa_tool = create_retrieval_tool(
    "./policies_v2/cont_edu_q.md",
    "cont_edu_qa_engine",
    "回答继续教育机构用户的相关问题，返回最相关的文档，如：",
    search_kwargs={"k": 3},
    chunk_size=100,
    separators=["\n\n"],
    use_cached_faiss=True,
)

individual_qa_agent_executor_v2 = create_atomic_retriever_agent(
    tools=[individual_qa_tool, RegistrationStatusToolIndividual()],
    qa_map_path="./policies_v2/individual_qa_map.json",
)
employing_unit_qa_agent_executor_v2 = create_atomic_retriever_agent(
    tools=[employing_unit_qa_tool, RegistrationStatusToolNonIndividual()],
    qa_map_path="./policies_v2/employing_unit_qa_map.json",
)
supervisory_department_qa_agent_executor_v2 = create_atomic_retriever_agent(
    tools=[supervisory_department_qa_tool, RegistrationStatusToolNonIndividual()],
    qa_map_path="./policies_v2/supervisory_dept_qa_map.json",
)
cont_edu_qa_agent_executor_v2 = create_atomic_retriever_agent(
    tools=[cont_edu_qa_tool, RegistrationStatusToolNonIndividual()],
    qa_map_path="./policies_v2/cont_edu_qa_map.json",
)


update_user_role_tools = [UpdateUserRoleTool2(), RegistrationStatusToolUniversal()]
update_user_role_agent = create_atomic_retriever_agent(
    tools=update_user_role_tools,
    system_prompt=f"""你是一个助手，可以使用以下工具。以下是每个工具的名称和描述：

        {render_text_description(update_user_role_tools)}
        
        ### 任务
        根据用户的输入 input, 你需要将用户意图分类为 `查询用户角色` 或者 `提供用户角色信息` 或者 `其他`。
        如果用户需要帮助查找他们的角色，请使用 {update_user_role_tools[1].name} 来搜索用户角色。
        如果用户的意图是提供他们的角色信息，请使用 {update_user_role_tools[0].name} 来更新用户角色。
        所有其他用户输入都应该被分类为 `其他`。不确定时，默认为`其他`。使用 {update_user_role_tools[0].name} 工具，将 'arguments' 中的 'user_role' 设置为 'unknown'。
        如果用户意图是`其他`，使用 {update_user_role_tools[0].name} 工具，将 'arguments' 中的 'user_role' 设置为 'unknown'。
        
        用户角色为：专技个人、用人单位、主管部门、继续教育机构、跳过
        注意：用户的问题可能包含角色，即使包含角色，用户的意图不一定是提供角色信息。因此，当包含角色时，你要更加小心的对用户的意图进行分类。
        注意：当用户意图查询信息是，用户不一定只会查询自己的角色，也可能查询其他信息。只有当用户查询角色或注册信息时，你才需要使用 {update_user_role_tools[1].name} 工具。否则，使用 {update_user_role_tools[0].name} 工具，将 'arguments' 中的 'user_role' 设置为 'unknown'。
        
        最终返回需要调用的工具名称和输入。返回的响应应该是一个 JSON 数据，其中包含 'name' 和 'arguments' 键。'argument' 的值应该是一个 json，其中包含要传递给工具的输入。

        ### 以下是一些示例：
        #### 查询用户角色:
        - "我想知道我的注册状态" -> 调用 {update_user_role_tools[1].name}, 将 'arguments' 中的 'user_id_number' 设置为 'unknown'。
        - "不知道啊，帮我查一下" -> 调用 {update_user_role_tools[1].name}, 将 'arguments' 中的 'user_id_number' 设置为 'unknown'。
        - "山东省济南市中心医院" -> 调用 {update_user_role_tools[1].name}, 将 'arguments' 中的 'user_id_number' 设置为 '山东省济南市中心医院'。
        - "济宁市人才服务中心" -> 调用 {update_user_role_tools[1].name}, 将 'arguments' 中的 'user_id_number' 设置为 '济宁市人才服务中心'。
        - "43942929391938222" -> 调用 {update_user_role_tools[1].name}, 将 'arguments' 中的 'user_id_number' 设置为 '43942929391938222'。
        
        #### 提供用户角色信息:
        - "我是专技个人" -> 调用 {update_user_role_tools[0].name}, 将 'arguments' 中的 'user_role' 设置为 'unknown'。
        - "专技个人" -> 调用 {update_user_role_tools[0].name}, 将 'arguments' 中的 'user_role' 设置为 '专技个人'。
        - "用人单位" -> 调用 {update_user_role_tools[0].name}, 将 'arguments' 中的 'user_role' 设置为 '用人单位'。
        - "主管部门" -> 调用 {update_user_role_tools[0].name}, 将 'arguments' 中的 'user_role' 设置为 '主管部门'。
        - "继续教育机构" -> 调用 {update_user_role_tools[0].name}, 将 'arguments' 中的 'user_role' 设置为 '继续教育机构'。
        - "跳过" -> 调用 {update_user_role_tools[0].name}, 将 'arguments' 中的 'user_role' 设置为 '跳过'。
        
        #### 其他
        - "继续教育机构如何注册" -> 调用 {update_user_role_tools[0].name}, 将 'arguments' 中的 'user_role' 设置为 'unknown'。
        - "注册如何审核" -> 调用 {update_user_role_tools[0].name}, 将 'arguments' 中的 'user_role' 设置为 'unknown'。
        - "专技个人注册如何审核" -> 调用 {update_user_role_tools[0].name}, 将 'arguments' 中的 'user_role' 设置为 'unknown'。
        - "单位怎么学时申报" -> 调用 {update_user_role_tools[0].name}, 将 'arguments' 中的 'user_role' 设置为 'unknown'。
        - "单位的培训计划怎么审核" -> 调用 {update_user_role_tools[0].name}, 将 'arguments' 中的 'user_role' 设置为 'unknown'。

        """,
    # TODO: why jining?
    qa_map_path="./policies_v2/jining_qa_map.json",
)


def check_role_qa_router(info):
    print(info["topic"])
    if "unknown" in info["topic"]["output"].lower():
        print("check role entering unknown")
        return update_user_role_agent
    elif "专技个人" in info["topic"]["output"].lower():
        print("entering 专技个人")
        return individual_qa_agent_executor_v2
    elif "用人单位" in info["topic"]["output"].lower():
        print("entering 用人单位")
        return employing_unit_qa_agent_executor_v2
    elif "主管部门" in info["topic"]["output"].lower():
        print("entering 主管部门")
        return supervisory_department_qa_agent_executor_v2
    elif "继续教育机构" in info["topic"]["output"].lower():
        print("entering 继续教育机构")
        return cont_edu_qa_agent_executor_v2
    print("默认进入专技个人")
    return individual_qa_agent_executor_v2


# ===========================================================================
#  START: MainChain - check user role
# ===========================================================================


class RegistrationStatusTool(BaseTool):
    """查询用户在大众云学平台上的注册状态"""

    name: str = "注册状态查询工具"
    description: str = (
        "用于查询用户在大众云学平台上的注册状态，需要指通过 json 指定用户身份证号 user_id_number "
    )
    return_direct: bool = True

    def _run(self, params) -> Any:
        print(params)
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

        return apis.get_registration_status(params_dict)


class UpdateUserRoleTool(BaseTool):
    """根据用户回答，更新用户角色"""

    name: str = "用户角色更新工具"
    description: str = (
        "用于更新用户在对话中的角色，需要指通过 json 指定用户角色 user_role "
    )
    return_direct: bool = True

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
        main_qa_agent_executor.agent.runnable.get_prompts()[0].template = (
            """Your ONLY job is to use a tool to answer the following question.

    You MUST use a tool to answer the question. 
    Simply Answer "您能提供更多关于这个问题的细节吗？" if you don't know the answer.
    DO NOT answer the question without using a tool.

    Current user role is """
            + user_role
            + """.

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
        # st.session_state.user_role = user_role
        return f"更新您的用户角色为{user_role}, 请问有什么可以帮到您？"


prompt = PromptTemplate.from_template(
    """Your ONLY job is to use a tool to answer the following question.

You MUST use a tool to answer the question. 
Simply Answer "您能提供更多关于这个问题的细节吗？" if you don't know the answer.
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
)
prompt.input_variables = [
    "agent_scratchpad",
    "input",
    "tool_names",
    "tools",
    "chat_history",
]

tools = [
    RegistrationStatusTool(),
    UpdateUserRoleTool(),
]

memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")
agent = create_react_agent(
    Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}), tools, prompt
)
main_qa_agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True
)


def check_user_role(inputs):
    template = main_qa_agent_executor.agent.runnable.get_prompts()[0].template.lower()
    start_index = template.find("current user role is") + len("current user role is")
    end_index = template.find("\n", start_index)
    result = template[start_index:end_index].strip()
    # result = st.session_state.get("user_role", "unknown")
    inputs["output"] = result
    return inputs


check_user_role_chain = RunnableLambda(check_user_role)
# ===========================================================================
#  END: MainChain - check user role
# ===========================================================================


qa_chain_v2 = {
    "topic": check_user_role_chain,
    "input": lambda x: x["input"],
} | RunnableLambda(check_role_qa_router)


# ===========================================================================
#  START: Login
# ===========================================================================
# 登录问题咨询
login_problems_detail_tool = create_retrieval_tool(
    "./policies_v2/login_problems_details_q.md",
    "login_problems_detail_engine",
    "回答用户登录问题的细节相关问题，返回最相关的文档，如：没有滑块，找不到滑块，登录为什么提示验证失败，哪里有滑块，密码错误，忘记密码，账号不存在，登录显示审核中",
    search_kwargs={"k": 3},
    chunk_size=100,
    separators=["\n\n"],
)


login_problem_classifier_prompt = PromptTemplate.from_template(
    """Given the user input AND chat history below, classify whether the conversation topic or user mentioned being about `没有滑块` or `密码错误` or `账号不存在` or `审核中` or `手机网页无法登录` or `页面不全` or `无法登录` or `验证失败`

# Do not answer the question. Simply classify it as being related to `没有滑块` or `密码错误` or `账号不存在` or `审核中` or `手机网页无法登录` or `页面不全` or `无法登录` or `验证失败`
# Do not respond with anything other than `没有滑块` or `密码错误` or `账号不存在` or `审核中` or `手机网页无法登录` or `页面不全` or `无法登录` or `验证失败`

{chat_history}
Question: {input}

# Classification:"""
)

login_problem_classifier_prompt.input_variables = ["input", "chat_history"]

login_problem_classifier_memory = ConversationBufferMemory(
    memory_key="chat_history", input_key="input"
)

login_problem_classifier_chain = LLMChain(
    llm=Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
    prompt=login_problem_classifier_prompt,
    memory=login_problem_classifier_memory,
    verbose=True,
)

# login_problem_agent_executor = create_react_agent_with_memory(
#     tools=[login_problems_detail_tool]
# )

login_problem_agent_executor = create_atomic_retriever_agent_single_tool_qa_map(
    login_problems_detail_tool,
    qa_map_path="./policies_v2/login_problems_details_qa_map.json",
)

login_problem_ask_user_executor = LLMChain(
    llm=Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
    prompt=PromptTemplate.from_template(
        """Your ONLY job is to say: 请问您无法登录或登录不上，提示是什么？

You may use different words to ask the same question. But do not answer anything other than asking the user to provide more information.
                                                                                                    
Begin!
"""
    ),
    verbose=True,
    output_key="output",
)


def login_problem_router(info):
    print(info["topic"])
    if "没有滑块" in info["topic"]["text"]:
        print("没有滑块")
        return login_problem_agent_executor
    elif "密码错误" in info["topic"]["text"]:
        print("密码错误")
        return login_problem_agent_executor
    elif "账号不存在" in info["topic"]["text"]:
        print("账号不存在")
        return login_problem_agent_executor
    elif "审核中" in info["topic"]["text"]:
        print("审核中")
        return login_problem_agent_executor
    elif "手机网页无法登录" in info["topic"]["text"]:
        print("手机网页无法登录")
        return login_problem_agent_executor
    elif "页面不全" in info["topic"]["text"]:
        print("页面不全")
        return login_problem_agent_executor
    elif "验证失败" in info["topic"]["text"]:
        print("页面不全")
        return login_problem_agent_executor
    elif "无法登录" in info["topic"]["text"]:
        print("无法登录")
        return login_problem_ask_user_executor

    return login_problem_ask_user_executor


login_problem_chain = {
    "topic": login_problem_classifier_chain,
    "input": lambda x: x["input"],
} | RunnableLambda(login_problem_router)

# ===========================================================================
#  END: Login
# ===========================================================================


# ===========================================================================
#  START: Forget Password
# ===========================================================================
# 忘记密码问题咨询
forgot_password_tool = create_retrieval_tool(
    "./policies_v2/forgot_password_q.md",
    "forgot_password_engine",
    "回答用户忘记密码的相关问题，返回最相关的文档，如：忘记密码怎么办，密码忘记了，找回密码，忘记密码手机号那里怎么是空的、手机号不显示、手机号怎么修改、手机号不用了，怎么找回、姓名或身份证号或所在单位有误、提示什么姓名错误、身份证号错误、所在单位有误、密码怎么保存不了、改密码怎么不行、改密码怎么保存不了、密码保存不了",
    search_kwargs={"k": 3},
    chunk_size=100,
    separators=["\n\n"],
)

forgot_password_classifier_prompt = PromptTemplate.from_template(
    """Given the user input AND chat history below, classify whether the conversation topic or user mentioned being about `忘记密码` or `找回密码` or `手机号` or `信息有误` or `保存不了` or `改密码怎么不行`

# Do not answer the question. Simply classify it as being related to `忘记密码` or `找回密码` or `手机号` or `信息有误` or `保存不了` or `改密码怎么不行`
# Do not respond with anything other than `忘记密码` or `找回密码` or `手机号` or `信息有误` or `保存不了` or `改密码怎么不行`

{chat_history}
Question: {input}

# Classification:"""
)

forgot_password_classifier_prompt.input_variables = ["input", "chat_history"]
forgot_password_classifier_memory = ConversationBufferMemory(
    memory_key="chat_history", input_key="input"
)
forgot_password_classifier_chain = LLMChain(
    llm=Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
    prompt=forgot_password_classifier_prompt,
    memory=forgot_password_classifier_memory,
    verbose=True,
)

# forgot_password_agent_executor = create_react_agent_with_memory(
#     tools=[forgot_password_tool]
# )

forgot_password_agent_executor = create_atomic_retriever_agent_single_tool_qa_map(
    forgot_password_tool, qa_map_path="./policies_v2/forgot_password_qa_map.json"
)

forgot_password_ask_user_executor = LLMChain(
    llm=Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
    prompt=PromptTemplate.from_template(
        """Your ONLY job is to say: 您可以在平台首页右侧——【登录】按钮右上方 ——点击【忘记密码？】找回密码。
do not answer anything other than asking the user to provide more information.
                                                                                                    
Begin!
"""
    ),
    verbose=True,
    output_key="output",
)


def forgot_password_router(info):
    print(info["topic"])
    if "忘记密码" in info["topic"]["text"]:
        print("忘记密码")
        return forgot_password_agent_executor
    elif "找回密码" in info["topic"]["text"]:
        print("找回密码")
        return forgot_password_agent_executor
    elif "手机号" in info["topic"]["text"]:
        print("手机号")
        return forgot_password_agent_executor
    elif "信息有误" in info["topic"]["text"]:
        print("信息有误")
        return forgot_password_agent_executor
    elif "保存不了" in info["topic"]["text"]:
        print("保存不了")
        return forgot_password_agent_executor
    elif "改密码怎么不行" in info["topic"]["text"]:
        print("改密码怎么不行")
        return forgot_password_agent_executor
    return forgot_password_ask_user_executor


forgot_password_chain = {
    "topic": forgot_password_classifier_chain,
    "input": lambda x: x["input"],
} | RunnableLambda(forgot_password_router)


# ===========================================================================
#  END: Forget Password
# ===========================================================================


# ===========================================================================
#  START: JINING
# ===========================================================================

# 济宁市
# jining_agent_executor = create_react_agent_with_memory(tools=[jn_city_tool])

jn_city_tool = create_retrieval_tool(
    "./policies_v2/jining_q.md",
    "jn_city_engine",
    "回答有关济宁市报班缴费，在线学习和缴费的相关问题，返回最相关的文档",
    search_kwargs={"k": 3},
    chunk_size=100,
    separators=["\n\n"],
)
jining_agent_executor = create_atomic_retriever_agent_single_tool_qa_map(
    jn_city_tool, qa_map_path="./policies_v2/jining_qa_map.json"
)

# ===========================================================================
#  END: JINING
# ===========================================================================


# ===========================================================================
#  START: CREDIT CHANIN
# ===========================================================================


class UpdateUserLocTool2(BaseTool):
    """根据用户回答，更新用户学习地市"""

    name: str = "用户学习地市更新工具"
    description: str = (
        "用于更新用户学习地市，需要指通过 json 指定用户学习地市 user_location "
    )
    # args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    def _run(self, params) -> Any:
        print(params)
        if isinstance(params, str):
            try:
                params_dict = json.loads(params)
            except json.JSONDecodeError:
                return (
                    "请问您是在哪个地市平台学习的？请先确认您的学习地市，以便我能为您提供相应的信息。我方负责的主要平台地市有：\n\n"
                    + LOC_STR
                )
        elif isinstance(params, dict):
            params_dict = params
        else:
            return (
                "请问您是在哪个地市平台学习的？请先确认您的学习地市，以便我能为您提供相应的信息。我方负责的主要平台地市有：\n\n"
                + LOC_STR
            )

        if params_dict is None:
            return (
                "请问您是在哪个地市平台学习的？请先确认您的学习地市，以便我能为您提供相应的信息。我方负责的主要平台地市有：\n\n"
                + LOC_STR
            )
        if "user_location" not in params_dict:
            return (
                "请问您是在哪个地市平台学习的？请先确认您的学习地市，以便我能为您提供相应的信息。我方负责的主要平台地市有：\n\n"
                + LOC_STR
            )
        user_location = params_dict["user_location"]
        if user_location is None:
            return (
                "请问您是在哪个地市平台学习的？请先确认您的学习地市，以便我能为您提供相应的信息。我方负责的主要平台地市有：\n\n"
                + LOC_STR
            )
        if user_location == "unknown":
            return (
                "请问您是在哪个地市平台学习的？请先确认您的学习地市，以便我能为您提供相应的信息。我方负责的主要平台地市有：\n\n"
                + LOC_STR
            )
        if user_location not in LOC_STR and user_location not in [
            "开放大学",
            "蟹壳云学",
            "专技知到",
            "文旅厅",
            "教师",
        ]:
            return (
                "请问您是在哪个地市平台学习的？请先确认您的学习地市，以便我能为您提供相应的信息。我方负责的主要平台地市有：\n\n"
                + LOC_STR
            )
        # if user_location not in LOC_STR:
        #     return "请问您是在哪个地市平台学习的？请先确认您的学习地市，以便我能为您提供相应的信息。我方负责的主要平台地市有：\n\n" + LOC_STR
        credit_problem_chain_executor.agent.runnable.get_prompts()[0].template = (
            """Use a tool to answer the user's qustion.

You MUST use a tool and generate a response based on tool's output.
DO NOT hallucinate!!!!
DO NOT Assume any user inputs. ALWAYS ask the user for more information if needed.
DO NOT Assume year, course_type, or user_id_number, ALWAYS ask if needed.

Note that you may need to translate user inputs. Here are a few examples for translating user inputs:
- user: "公需", output: "公需课"
- user: "公", output: "公需课"
- user: "专业", output: "专业课"
- user: "专", output: "专业课"
- user: "19年", output: "2019"
- user: "19", output: "2019"
- user: "2019年”, output: "2019"

user location: """
            + user_location
            + """
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
        # st.session_state.user_role = user_role
        return f"谢谢，已为您更新您的学习地市为{user_location}, 现在请您提供身份证号码，以便我查询您的学时状态。"


class CheckUserCreditTool(BaseTool):
    """根据用户回答，检查用户学时状态"""

    name: str = "检查用户学时状态工具"
    description: str = (
        "用于检查用户学时状态，需要指通过 json 指定用户身份证号 user_id_number、用户想要查询的年份 year、用户想要查询的课程类型 course_type "
    )
    # args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    def _run(self, params) -> Any:

        params = params.replace("'", '"')
        print(params, type(params))
        CONTEXT_PROMPT = "You must ask the human about {context}. Reply with schema #2."
        try:
            params_dict = json.loads(params)
        except json.JSONDecodeError as e:
            print(e)
            return "麻烦您提供一下您的身份证号，我这边帮您查一下"

        if "user_id_number" not in params_dict:
            return "麻烦您提供一下您的身份证号"
        if isinstance(params_dict["user_id_number"], list):
            params_dict["user_id_number"] = params_dict["user_id_number"][0]
        if params_dict["user_id_number"] is None:
            return "麻烦您提供一下您的身份证号"
        if len(params_dict["user_id_number"]) < 2:
            return "身份证号似乎不太对，麻烦您提供一下您正确的身份证号"

        if "year" not in params_dict:
            return "您问的是哪个年度的课程？如：2019年"
        if isinstance(params_dict["year"], list):
            params_dict["year"] = params_dict["year"][0]
        if params_dict["year"] is None:
            return "您问的是哪个年度的课程？如：2019年"
        if len(str(params_dict["year"])) < 2:
            return "年度似乎不太对，麻烦您确认你的课程年度。如：2019年"

        if "course_type" not in params_dict:
            return "您要查询的是公需课还是专业课"
        if isinstance(params_dict["course_type"], list):
            params_dict["course_type"] = params_dict["course_type"][0]
        if params_dict["course_type"] is None:
            return "您要查询的是公需课还是专业课"
        if len(params_dict["course_type"]) < 2:
            return "请确认您要查询的是公需课还是专业课"

        user_id_number = str(params_dict["user_id_number"])
        year = re.search(r"\d+", str(params_dict["year"])).group()
        course_type = str(params_dict["course_type"])

        template = credit_problem_chain_executor.agent.runnable.get_prompts()[
            0
        ].template.lower()
        # print(template)
        start_index = template.find("user location: ") + len("user location: ")
        end_index = template.find("\n", start_index)
        user_provided_loc = template[start_index:end_index].strip()

        user_loc = REGISTRATION_STATUS[user_id_number]["注册地点"]

        match_location = check_user_location(user_provided_loc, [user_loc])
        if match_location is None:
            match_other_loc = check_user_location(
                user_provided_loc,
                [
                    "开放大学",
                    "蟹壳云学",
                    "专技知到",
                    "文旅厅",
                    "教师",
                ],
            )
            if match_other_loc is not None:
                if match_other_loc == "文旅厅":
                    return "本平台只是接收方，学时如果和您实际不符，建议您先咨询您的学习培训平台，学时是否有正常推送过来，只有推送了我们才能收到，才会显示对应学时。"
                return f"经查询您本平台的单位所在区域是{user_loc}，不是省直，非省直单位学时无法对接。"
            return f"经查询您本平台的单位所在区域是{user_loc}，不是{user_provided_loc}，区域不符学时无法对接，建议您先进行“单位调转”,调转到您所在的地市后，再联系您的学习培训平台，推送学时。"
        else:
            match_other_loc = check_user_location(
                user_provided_loc,
                [
                    "开放大学",
                    "蟹壳云学",
                    "专技知到",
                    "文旅厅",
                    "教师",
                ],
            )
            if match_other_loc is not None:
                return "请先咨询您具体的学习培训平台，学时是否有正常推送过来，只有推送了我们才能收到，才会显示对应学时。"
            hours = CREDIT_HOURS.get(user_id_number)
            if hours is None:
                return "经查询，平台还未接收到您的任何学时信息，建议您先咨询您的学习培训平台，学时是否全部推送，如果已确定有推送，请您24小时及时查看对接情况；每年7月至9月，因学时对接数据较大，此阶段建议1-3天及时关注。"
            year_hours = hours.get(year)
            if year_hours is None:
                return f"经查询，平台还未接收到您在{year}年度的任何学时信息，建议您先咨询您的学习培训平台，学时是否全部推送，如果已确定有推送，请您24小时及时查看对接情况；每年7月至9月，因学时对接数据较大，此阶段建议1-3天及时关注。"
            course_year_hours = year_hours.get(course_type)
            if course_year_hours is None:
                return f"经查询，平台还未接收到您在{year}年度{course_type}的学时信息，建议您先咨询您的学习培训平台，学时是否全部推送，如果已确定有推送，请您24小时及时查看对接情况；每年7月至9月，因学时对接数据较大，此阶段建议1-3天及时关注。"
            if len(course_year_hours) == 0:
                return f"经查询，平台还未接收到您在{year}年度{course_type}的学时信息，建议您先咨询您的学习培训平台，学时是否全部推送，如果已确定有推送，请您24小时及时查看对接情况；每年7月至9月，因学时对接数据较大，此阶段建议1-3天及时关注。"
            total_hours = sum([x["学时"] for x in course_year_hours])
            finished_hours = sum(
                [
                    x["学时"]
                    for x in course_year_hours
                    if x["进度"] == 100 and x["考核"] == "合格"
                ]
            )
            unfinished_courses = [
                f"{x['课程名称']}完成了{x['进度']}%"
                for x in course_year_hours
                if x["进度"] < 100
            ]
            untested_courses = [
                x["课程名称"] for x in course_year_hours if x["考核"] == "未完成"
            ]
            unfinished_str = "  \n\n".join(unfinished_courses)
            untested_str = "  \n\n".join(untested_courses)

            res_str = f"经查询，您在{year}年度{course_type}的学时情况如下：  \n\n"
            res_str += f"您报名的总学时：{total_hours}  \n\n"
            res_str += f"已完成学时：{finished_hours}  \n\n"
            res_str += f"其中，以下几节课进度还没有达到100%，每节课进度看到100%后才能计入学时  \n\n"
            res_str += unfinished_str + "  \n\n"
            res_str += f"以下几节课还没有完成考试，考试通过后才能计入学时  \n\n"
            res_str += untested_str + "  \n\n"
            return res_str


credit_problem_prompt = PromptTemplate.from_template(
    """Use a tool to answer the user's qustion.

You MUST use a tool and generate a response based on tool's output.
DO NOT hallucinate!!!! 
DO NOT Assume any user inputs. ALWAYS ask the user for more information if needed.
DO NOT Assume year, course_type, or user_id_number, ALWAYS ask if needed.

Note that you may need to translate user inputs. Here are a few examples for translating user inputs:
- user: "公需", output: "公需课"
- user: "公", output: "公需课"
- user: "专业", output: "专业课"
- user: "专", output: "专业课"
- user: "19年", output: "2019"
- user: "19", output: "2019"
- user: "2019年”, output: "2019"


user location: unknown

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
credit_problem_prompt.input_variables = [
    "agent_scratchpad",
    "input",
    "tool_names",
    "tools",
    "chat_history",
]

credit_problem_tools = [CheckUserCreditTool()]
credit_problem_memory = ConversationBufferMemory(
    memory_key="chat_history", input_key="input"
)
credit_problem_chain = create_react_agent(
    Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
    # ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.3),
    credit_problem_tools,
    credit_problem_prompt,
)
credit_problem_chain_executor = AgentExecutor.from_agent_and_tools(
    agent=credit_problem_chain,
    tools=credit_problem_tools,
    memory=credit_problem_memory,
    verbose=True,
    handle_parsing_errors=True,
)

# update user location agent
update_user_location_agent = create_single_function_call_agent(UpdateUserLocTool2())


def check_user_loc_and_route(info):
    print(info["topic"])
    if "unknown" in info["topic"]["output"].lower():
        print("entering update_user_location_agent")
        return update_user_location_agent
    print("entering credit_problem_chain")
    return credit_problem_chain_executor
    # return main_credit_problem_agent


def check_user_loc(inputs):
    template = credit_problem_chain_executor.agent.runnable.get_prompts()[
        0
    ].template.lower()
    # print(template)
    start_index = template.find("user location: ") + len("user location: ")
    end_index = template.find("\n", start_index)
    result = template[start_index:end_index].strip()
    # result = st.session_state.get("user_role", "unknown")
    inputs["output"] = result
    return inputs


check_loc_chain = RunnableLambda(check_user_loc)

main_credit_problem_chain = {
    "topic": check_loc_chain,
    "input": lambda x: x["input"],
} | RunnableLambda(check_user_loc_and_route)

# ===========================================================================
#  END: CREDIT CHANIN
# ===========================================================================


# ===========================================================================
#  START: Courese Progress
# ===========================================================================

# course progress
course_progress_problems_prompt = PromptTemplate.from_template(
    """Answer the user's question step by step. Don't give the whole answer at once. Guide the user to the solution.

Always start with Step 1 below, DO NOT go to Step 2. Only execute Step 1 first. Do Not include the keyword `Step 1` or `Step 2` in your response.

Step 1. First check the user's learning method belongs to 电脑浏览器 or 手机微信扫码

Step 2. Based on the user's choice in Step 1,
If the user's learning method belongs to 电脑浏览器 or 手机微信扫码, then say 电脑浏览器请不要使用IE、edge等自带浏览器，可以使用搜狗、谷歌、360浏览器极速模式等浏览器试试。
Otherwise, say 目前支持的学习方式是电脑浏览器或者手机微信扫码两种，建议您再使用正确的方式试试
If the user's used the right method but still has problems, then say 建议清除浏览器或者微信缓存再试试
If the user used the right method and 清除了缓存, then say，抱歉，您的问题涉及到测试，建议您联系平台的人工热线客服或者在线客服进行反馈

Answer the question in Chinese.

{chat_history}
Question: {input}
"""
)
course_progress_problems_prompt.input_variables = [
    "input",
    "chat_history",
]
course_progress_problems_llm = Tongyi(
    model_name="qwen-max", model_kwargs={"temperature": 0.3}
)
course_progress_problems_memory = ConversationBufferMemory(
    memory_key="chat_history", input_key="input", return_messages=True
)
course_progress_problems_llm_chain = LLMChain(
    llm=course_progress_problems_llm,
    memory=course_progress_problems_memory,
    prompt=course_progress_problems_prompt,
    verbose=True,
    output_key="output",
)

# ===========================================================================
#  END: Courese Progress
# ===========================================================================


# ===========================================================================
#  START: Multiple Login
# ===========================================================================

# multiple login
multiple_login_prompt = PromptTemplate.from_template(
    """Answer the user's question step by step. Don't give the whole answer at once. Guide the user to the solution.

Always start with Step 1 below, DO NOT go to Step 2. Only execute Step 1 first. Do Not include the keyword `Step 1` or `Step 2` in your response.

Step 1
First check the user's learning method belongs to 电脑浏览器 or 手机微信扫码

Step 2
Based on the user's choice in Step 1,
If the user's learning method belongs to 电脑浏览器 or 手机微信扫码, then say 请勿使用电脑和手机同时登录账号学习，也不要使用电脑或手机同时登录多人账号学习。
If the user say 没有登录多个账号/没有同时登录 etc., say 建议您清除电脑浏览器或手机微信缓存，并修改平台登录密码后重新登录学习试试。

Answer the question in Chinese.

{chat_history}
Question: {input}
"""
)
multiple_login_prompt.input_variables = [
    "input",
    "chat_history",
]
multiple_login_llm = Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3})
multiple_login_memory = ConversationBufferMemory(
    memory_key="chat_history", input_key="input", return_messages=True
)
multiple_login_llm_chain = LLMChain(
    llm=multiple_login_llm,
    memory=multiple_login_memory,
    prompt=multiple_login_prompt,
    verbose=True,
    output_key="output",
)

# ===========================================================================
#  END: Multiple Login
# ===========================================================================


# ================================================================================
# START: Refund
# ================================================================================


class RefundTool(BaseTool):
    """根据用户回答，检查用户购买课程记录"""

    name: str = "检查用户购买课程记录工具"
    description: str = (
        "用于检查用户购买课程记录，需要指通过 json 指定用户身份证号 user_id_number、用户想要查询的课程年份 year、用户想要查询的课程名称 course_name "
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
            return "麻烦您提供一下您的身份证号，我这边帮您查一下"

        if "user_id_number" not in params_dict:
            return "麻烦您提供一下您的身份证号"
        if params_dict["user_id_number"] is None:
            return "麻烦您提供一下您的身份证号"
        if len(params_dict["user_id_number"]) < 2:
            return "麻烦您提供一下您正确的身份证号"

        if "year" not in params_dict:
            return "您问的是哪个年度的课程？如：2019年"
        if params_dict["year"] is None:
            return "您问的是哪个年度的课程？如：2019年"
        if len(params_dict["year"]) < 4:
            return "您问的是哪个年度的课程？如：2019年"

        if "course_name" not in params_dict:
            return "您问的课程名称是什么？如：新闻专业课培训班"
        if params_dict["course_name"] is None:
            return "您问的课程名称是什么？如：新闻专业课培训班"
        if len(params_dict["course_name"]) < 2:
            return "您问的课程名称是什么？如：新闻专业课培训班"

        user_id_number = params_dict["user_id_number"]

        year = params_dict["year"]
        year = re.search(r"\d+", year).group()

        course_name = params_dict["course_name"]
        if COURSE_PURCHASES.get(user_id_number) is not None:
            purchases = COURSE_PURCHASES.get(user_id_number)
            if year in purchases:
                if course_name in purchases[year]:
                    progress = purchases[year][course_name]["进度"]
                    if progress == 0:
                        return "经查询您的这个课程没有学习，您可以点击右上方【我的学习】，选择【我的订单】，找到对应课程点击【申请售后】，费用在1个工作日会原路退回。"
                    return f"经查询，您的课程{course_name}学习进度为{progress}%，可以按照未学的比例退费，如需退费请联系平台的人工热线客服或者在线客服进行反馈。"
                return f"经查询，您在{year}年度，没有购买{course_name}，请您确认您的课程名称、年度、身份证号是否正确。"
            return f"经查询，您在{year}年度，没有购买{course_name}，请您确认您的课程名称、年度、身份证号是否正确。"
        return f"经查询，您在{year}年度，没有购买{course_name}，请您确认您的课程名称、年度、身份证号是否正确。"


refund_prompt = PromptTemplate.from_template(
    """Use a tool to answer the user's qustion.

Ask the user to provide 身份证号，in order to 查询课程信息
You MUST use your ONLY tool to answer the user question.

When user input a number longer than 6 digits, use it as user 身份证号 in the context for the tool.
DO NOT hallucinate!!!! DO NOT Assume any user inputs. ALWAYS ask the user for more information if needed.

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

Answer the question in Chinese.

Begin!

{chat_history}
Question: {input}
Thought:{agent_scratchpad}
"""
)

refund_prompt.input_variables = [
    "agent_scratchpad",
    "input",
    "tool_names",
    "tools",
    "chat_history",
]

refund_tools = [RefundTool()]
refund_memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")
refund_chain = create_react_agent(
    Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
    # ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.3),
    refund_tools,
    refund_prompt,
)
refund_chain_executor = AgentExecutor.from_agent_and_tools(
    agent=refund_chain,
    tools=refund_tools,
    memory=refund_memory,
    verbose=True,
    handle_parsing_errors=True,
)


# ================================================================================
# END: Refund
# ================================================================================

# ================================================================================
# START: Can't find Course
# ================================================================================


class CheckPurchaseTool(BaseTool):
    """根据用户回答，检查用户购买课程记录"""

    name: str = "检查用户购买课程记录工具"
    description: str = (
        "用于检查用户购买课程记录，需要指通过 json 指定用户身份证号 user_id_number、用户想要查询的课程年份 year、用户想要查询的课程名称 course_name "
    )
    # args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    def _run(self, params) -> Any:

        params = params.replace("'", '"')
        print(params, type(params))
        try:
            params_dict = json.loads(params)
            params_dict = {k: str(v) for k, v in params_dict.items()}
        except json.JSONDecodeError as e:
            print(e)
            return "麻烦您提供一下您的身份证号，我这边帮您查一下"

        if "user_id_number" not in params_dict:
            return "麻烦您提供一下您的身份证号"
        if params_dict["user_id_number"] is None:
            return "麻烦您提供一下您的身份证号"
        if len(str(params_dict["user_id_number"])) < 2:
            return "麻烦您提供一下您正确的身份证号"

        if "year" not in params_dict:
            return "您问的是哪个年度的课程？如：2019年"
        if params_dict["year"] is None:
            return "您问的是哪个年度的课程？如：2019年"
        if len(str(params_dict["year"])) < 4:
            return "麻烦您确认你的课程年度。如：2019年"

        if "course_name" not in params_dict:
            return "您问的课程名称是什么？如：新闻专业课培训班"
        if params_dict["course_name"] is None:
            return "您问的课程名称是什么？如：新闻专业课培训班"
        if len(params_dict["course_name"]) < 2:
            return "请您提供您想要查询的课程的正确名称。如：新闻专业课培训班"

        user_id_number = params_dict["user_id_number"]

        year = params_dict["year"]
        year = re.search(r"\d+", year).group()

        course_name = params_dict["course_name"]
        if COURSE_PURCHASES.get(user_id_number) is not None:
            purchases = COURSE_PURCHASES.get(user_id_number)
            if year in purchases:
                if course_name in purchases[year]:
                    progress = purchases[year][course_name]["进度"]
                    if progress == 0:
                        return f"经查询，您已经购买{year}年度的{course_name}，请前往专业课平台，点击右上方【我的学习】找到对应课程直接学习。"
                    return f"经查询，您已经购买{year}年度的{course_name}，您的学习进度为{progress}%。请前往专业课平台，点击右上方【我的学习】找到对应课程继续学习。"
                return f"经查询，您在{year}年度，没有购买{course_name}，请您确认您的课程名称、年度、身份证号是否正确。"
            return f"经查询，您在{year}年度，没有购买{course_name}，请您确认您的课程名称、年度、身份证号是否正确。"
        return f"经查询，您在{year}年度，没有购买{course_name}，请您确认您的课程名称、年度、身份证号是否正确。"


cannot_find_course_prompt = PromptTemplate.from_template(
    """Use a tool to answer the user's qustion.

Ask the user to provide 身份证号，in order to 检查用户购买课程记录
You MUST use a tool and generate a response based on tool's output.

When user input a number longer than 6 digits, use it as user 身份证号 in the context for the tool.
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

Answer the question in Chinese.

{chat_history}
Question: {input}
Thought:{agent_scratchpad}
"""
)
cannot_find_course_prompt.input_variables = [
    "agent_scratchpad",
    "input",
    "tool_names",
    "tools",
    "chat_history",
]

cannot_find_course_tools = [CheckPurchaseTool()]
cannot_find_course_memory = ConversationBufferMemory(
    memory_key="chat_history", input_key="input"
)
cannot_find_course_chain = create_react_agent(
    Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
    # ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.3),
    cannot_find_course_tools,
    cannot_find_course_prompt,
)
cannot_find_course_chain_executor = AgentExecutor.from_agent_and_tools(
    agent=cannot_find_course_chain,
    tools=cannot_find_course_tools,
    memory=cannot_find_course_memory,
    verbose=True,
    handle_parsing_errors=True,
)

# ================================================================================
# END: Can't find Course
# ================================================================================


# MAIN ENTRY POINT


main_question_classifier_template = """根据用户的输入 input 以及对话历史记录 chat_history，判定用户问的内容属于以下哪一类： `学时没显示` 或者 `学时有问题` 或者 `济宁市：如何报班、报名` 或者 `济宁市：课程进度不对` 或者 `济宁市：多个设备，其他地方登录` 或者 `济宁市：课程退款退费，课程买错了` 或者 `济宁市：课程找不到，课程没有了` 或者 `无法登录` 或者 `忘记密码` 或者 `找回密码` 或者 `济宁市` 或者 `注册` 或者 `审核` 或者 `学时对接` 或者 `学时申报` 或者 `学时审核` 或者 `系统操作` 或者 `修改信息` 或者 `其他`.

# 不要回答用户的问题。仅把用户的问题归类为 `学时没显示` 或 `学时有问题` 或 `济宁市：课程进度不对` 或 `济宁市：多个设备，其他地方登录` 或 `济宁市：课程退款退费，课程买错了` 或 `济宁市：课程找不到，课程没有了` 或 `无法登录` 或 `忘记密码` 或 `找回密码` 或 `济宁市` 或 `注册` 或 `审核` 或 `学时对接` 或 `学时申报` 或 `学时审核` 或 `系统操作` 或 `修改信息` 或 `其他`.
# 不要回答除此之外的任何内容。

以下是一些例子：
- "学时没显示" -> 分类为 `学时没显示`
- "学时有问题" -> 分类为 `学时有问题`
- "学时没对接" -> 分类为 `学时没显示`
- "学时没对接" -> 分类为 `学时没显示`
- "学时不对接" -> 分类为 `学时没显示`
- "学时不对" -> 分类为 `学时有问题`
- "学时对接" -> 分类为 `学时对接`
- "学时报错了" -> 分类为 `学时申报`
- "账号密码是什么" -> 分类为 `注册`
- "学时申报" -> 分类为 `学时申报`
- "济宁市，如何补学" -> 分类为 `济宁市`

注意：
如果用户提到了 "济宁市"，你应该将其分类为与 `济宁市` 相关。如果用户没有提到 "济宁市"，你绝对不能将其分类为与 `济宁市` 相关。
如果用户提到了以下具体的济宁市问题，你应该将其分类到济宁市具体的问题中。其他具体问题，统一归类为 `济宁市`。
- "济宁市：课程进度不对" -> 分类为 `济宁市：课程进度不对`
- "济宁市：多个设备，其他地方登录" -> 分类为 `济宁市：多个设备，其他地方登录`
- "济宁市：课程退款退费，课程买错了" -> 分类为 `济宁市：课程退款退费，课程买错了`
- "济宁市：课程找不到，课程没有了" -> 分类为 `济宁市：课程找不到，课程没有了`

{chat_history}
Question: {input}

Classification:"""

main_question_classifier_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template=main_question_classifier_template,
)

main_question_classifier_mem = ConversationBufferMemory(
    memory_key="chat_history", input_key="input"
)

main_question_classifier = LLMChain(
    llm=Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
    prompt=main_question_classifier_prompt,
    memory=main_question_classifier_mem,
    verbose=True,
)


if "topic" not in st.session_state:
    st.session_state.topic = None


def main_question_classifier_and_route(info):
    print(info)
    if st.session_state.topic is None:
        st.session_state.topic = info["topic"]["text"]
    else:
        info["topic"]["text"] = st.session_state.topic
    if "学时没显示" in info["topic"]["text"]:
        print("学时没显示")
        return main_credit_problem_chain
        # return test_chain
    if "学时有问题" in info["topic"]["text"]:
        print("学时有问题")
        return main_credit_problem_chain
    # if "济宁市：如何报班、报名" in info["topic"]["text"]:
    #     print("济宁市：如何报班、报名")
    #     return register_class_llm_chain

    if "济宁市：课程进度不对" in info["topic"]["text"]:
        print("济宁市：课程进度不对")
        return course_progress_problems_llm_chain
    if "济宁市：多个设备，其他地方登录" in info["topic"]["text"]:
        print("济宁市：多个设备，其他地方登录")
        return multiple_login_llm_chain
    if "济宁市：课程退款退费，课程买错了" in info["topic"]["text"]:
        print("济宁市：课程退款退费，课程买错了")
        return refund_chain_executor
        # return refund_full_chain
    if "济宁市：课程找不到，课程没有了" in info["topic"]["text"]:
        print("济宁市：课程找不到，课程没有了")
        return cannot_find_course_chain_executor

    if "济宁市" in info["topic"]["text"]:
        print("济宁市")
        return jining_agent_executor  # TODO: Add the chain for Jinin city

    # 无法登录咨询
    if "无法登录" in info["topic"]["text"]:
        print("无法登录")
        return login_problem_chain

    # 忘记密码咨询
    if "忘记密码" in info["topic"]["text"]:
        print("忘记密码")
        return forgot_password_chain
    if "找回密码" in info["topic"]["text"]:
        print("找回密码")
        return forgot_password_chain

    if "其他" in info["topic"]["text"]:
        print("other")
        return qa_chain_v2

    if "注册" in info["topic"]["text"]:
        print("注册")
        return qa_chain_v2
    if "审核" in info["topic"]["text"]:
        print("审核")
        return qa_chain_v2
    if "学时对接" in info["topic"]["text"]:
        print("学时对接")
        return qa_chain_v2
    if "系统操作" in info["topic"]["text"]:
        print("系统操作")
        return qa_chain_v2
    if "修改信息" in info["topic"]["text"]:
        print("修改信息")
        return qa_chain_v2
    if "其他" in info["topic"]["text"]:
        print("其他")
        return qa_chain_v2
    if "学时申报" in info["topic"]["text"]:
        print("学时申报")
        return qa_chain_v2
    if "学时审核" in info["topic"]["text"]:
        print("学时审核")
        return qa_chain_v2

    print("unknown")
    return qa_chain_v2


full_chain = {
    "topic": main_question_classifier,
    "input": lambda x: x["input"],
} | RunnableLambda(main_question_classifier_and_route)

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = full_chain

if prompt := st.chat_input(
    "您的问题"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("正在输入..."):
            response = st.session_state.chat_engine.invoke({"input": prompt})
            print(response)
            st.write(response["output"])
            message = {"role": "assistant", "content": response["output"]}
            st.session_state.messages.append(message)  # Add response to message history
