import json
import os
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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda

from statics import COURSE_PURCHASES, CREDIT_HOURS, LOC_STR, REGISTRATION_STATUS
import re

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings


os.environ["DASHSCOPE_API_KEY"] = "sk-91ee79b5f5cd4838a3f1747b4ff0e850"
# os.environ["DASHSCOPE_API_KEY"] = "sk-c92ed98926194b84a41a73db62af31d5"
os.environ["OPENAI_API_KEY"] = "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"
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
            "content": "欢迎您来到大众云学，我是大众云学的专家助手，我可以回答关于大众云学的所有问题。测试请使用身份证号372323199509260348。测试公需课/专业课学时，请使用年份2019/2020。测试课程购买，退款等，请使用年份2023，课程名称新闻专业课培训班。",
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
        main_qa_agent_executor.agent.runnable.get_prompts()[0].template = (
            """Your ONLY job is to use a tool to answer the following question.

You MUST use a tool to answer the question. 
Simply Answer "抱歉，根据我的搜索结果，我无法回答这个问题" if you don't know the answer.
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


class CheckUserRoleTool(BaseTool):
    """根据用户回答，检查用户角色"""

    name: str = "检查用户角色工具"
    description: str = "用于检查用户在对话中的角色，无需输入参数 "
    # args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    def _run(self, params) -> Any:
        print(params)
        template = main_qa_agent_executor.agent.runnable.get_prompts()[
            0
        ].template.lower()
        # print(template)
        start_index = template.find("current user role is") + len(
            "current user role is"
        )
        end_index = template.find("\n", start_index)
        result = template[start_index:end_index].strip()
        # result = st.session_state.get("user_role", "unknown")
        return result


class CheckUserLocTool(BaseTool):
    """根据用户回答，检查用户学习的地市"""

    name: str = "检查用户地市工具"
    description: str = "用于检查用户地市，无需输入参数 "
    # args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    def _run(self, params) -> Any:
        print(params)
        template = credit_problem_chain_executor.agent.runnable.get_prompts()[
            0
        ].template.lower()
        # print(template)
        start_index = template.find("user location: ") + len("user location: ")
        end_index = template.find("\n", start_index)
        result = template[start_index:end_index].strip()
        # result = st.session_state.get("user_role", "unknown")
        return result


class UpdateUserLocTool(BaseTool):
    """根据用户回答，更新用户学习地市"""

    name: str = "用户学习地市更新工具"
    description: str = (
        "用于更新用户学习地市，需要指通过 json 指定用户学习地市 user_location "
    )
    # args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    # def _run(self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None) -> Any:
    def _run(self, params) -> Any:
        print(params)
        try:
            params_dict = json.loads(params)
        except json.JSONDecodeError:
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
        # if user_location not in LOC_STR:
        #     return "请问您是在哪个地市平台学习的？请先确认您的学习地市，以便我能为您提供相应的信息。我方负责的主要平台地市有：\n\n" + LOC_STR
        credit_problem_chain_executor.agent.runnable.get_prompts()[0].template = (
            """Use a tool to answer the user's qustion.

You MUST use a tool and generate a response based on tool's output.
When user input a number longer than 6 digits, use it as user id number in the context for the tool.
When the user input a four-digit number, use it as year in the context for the tool.
DO NOT hallucinate!!!!
                                                     
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
        if len(params_dict["user_id_number"]) < 2:
            return "身份证号似乎不太对，麻烦您提供一下您正确的身份证号"
        if "year" not in params_dict:
            return "您问的是哪个年度的课程？如：2019年"
        if len(str(params_dict["year"])) < 2:
            return "年度似乎不太对，麻烦您确认你的课程年度。如：2019年"
        if "course_type" not in params_dict:
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

        if user_provided_loc not in user_loc and user_loc not in user_provided_loc:
            if user_provided_loc in ["开放大学","蟹壳云学","专技知到","文旅厅","教师"]:
                return f"经查询您本平台的单位所在区域是{user_loc}，不是省直，非省直单位学时无法对接。"
            return f"经查询您本平台的单位所在区域是{user_loc}，不是{user_provided_loc}，区域不符学时无法对接，建议您先进行“单位调转”,调转到您所在的地市后，再联系您的学习培训平台，推送学时。"
        else:
            if user_provided_loc in ["开放大学","蟹壳云学","专技知到","文旅厅","教师"]:
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
            finished_hours = sum([x["学时"] for x in course_year_hours if x["进度"] == 100 and x["考核"] == "合格"])
            unfinished_courses = [f"{x['课程名称']}完成了{x['进度']}%" for x in course_year_hours if x["进度"] < 100]
            untested_courses = [x['课程名称'] for x in course_year_hours if x["考核"] == "未完成"]
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
        if len(params_dict["user_id_number"]) < 2:
            return "身份证号似乎不太对，麻烦您提供一下您正确的身份证号"
        if "year" not in params_dict:
            return "您问的是哪个年度的课程？如：2019年"
        if len(params_dict["year"]) < 4:
            return "年度似乎不太对，麻烦您确认你的课程年度。如：2019年"
        if "course_name" not in params_dict:
            return "您问的课程名称是什么？如：新闻专业课培训班"
        if len(params_dict["course_name"]) < 2:
            return "课程名称似乎不太对，请您提供您想要查询的课程的正确名称。如：新闻专业课培训班"

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
        if len(str(params_dict["user_id_number"])) < 2:
            return "身份证号似乎不太对，麻烦您提供一下您正确的身份证号"
        if "year" not in params_dict:
            return "您问的是哪个年度的课程？如：2019年"
        if len(str(params_dict["year"])) < 4:
            return "年度似乎不太对，麻烦您确认你的课程年度。如：2019年"
        if "course_name" not in params_dict:
            return "您问的课程名称是什么？如：新闻专业课培训班"
        if len(params_dict["course_type"]) < 2:
            return "课程名称似乎不太对，请您提供您想要查询的课程的正确名称。如：新闻专业课培训班"

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
):
    # Load files
    loader = UnstructuredMarkdownLoader(markdown_path)
    docs = loader.load()
    print(docs)

    # Declare the embedding model
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
    )
    # embeddings = OpenAIEmbeddings()

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

faq_employing_unit_tool = create_retrieval_tool(
    "./policies/registration/employing_unit_reg_page_faq.md",
    "employing_unit_registration_faq_engine",
    "回答用人单位注册页面细项，以及常见问题，返回最相关的文档，如：单位性质和级别怎么选，单位所属行业选什么，主管部门怎么选/什么意思、上级单位是什么意思/怎么选，同级人社选什么，信息选错了怎么办",
)

faq_cont_edu_tool = create_retrieval_tool(
    "./policies/registration/continuing_edu_inst_reg_page_faq.md",
    "continuing_education_institute_registration_faq_engine",
    "回答继续教育机构注册页面细项，以及常见问题，返回最相关的文档，如：机构级别怎么选、什么意思，行业主管部门是什么意思、怎么选，同级人社部门怎么选/同级人社呢，选错了怎么办/选的不对会有什么影响",
)

cannot_register_tool = create_retrieval_tool(
    "./policies/registration/cannot_register.md",
    "cannot_register_engine",
    "回答用户无法注册的相关问题，返回最相关的文档，如：注册不了怎么办，注册不上怎么办，注册不了，无法注册，注册保存以后什么反应也没有，注册没反应",
    search_kwargs={"k": 1},
)

login_problems_tool = create_retrieval_tool(
    "./policies/registration/login_problems.md",
    "login_problems_engine",
    "回答用户登录问题的相关问题，返回最相关的文档，如：登录不了、无法登录、怎么登录不上",
    search_kwargs={"k": 1},
    chunk_size=100,
    separators=["\n\n"],
)

login_problems_detail_tool = create_retrieval_tool(
    "./policies/registration/login_problems_details.md",
    "login_problems_detail_engine",
    "回答用户登录问题的细节相关问题，返回最相关的文档，如：没有滑块，找不到滑块，登录为什么提示验证失败，哪里有滑块，密码错误，忘记密码，账号不存在，登录显示审核中",
    search_kwargs={"k": 1},
    chunk_size=100,
    separators=["\n\n"],
)

# TODO: Add more here
forgot_password_tool = create_retrieval_tool(
    "./policies/registration/forgot_password.md",
    "forgot_password_engine",
    "回答用户忘记密码的相关问题，返回最相关的文档，如：忘记密码怎么办，密码忘记了，找回密码",
    # search_kwargs={"k": 1},
    # chunk_size=100,
    # separators=["\n\n"]
)

# create operation retrievers
individual_operation_tool = create_retrieval_tool(
    "./policies/operation/individual_operation.md",
    "individual_operation_engine",
    "回答专技个人学时、学时申报、修改单位的系统操作相关问题，返回最相关的文档，如：怎么学时申报，如何提交学时，为什么不能学时申报，学时申报信息天填错了怎么办，学时信息填好后无法保存，我怎么不能学时申报、我的账号里怎么没有学时申报，证书和发明专利能申报、抵扣多少学时。再如：怎么修改单位，修改单位的话，现在单位能知道吗，现在单位审核吗，单位调转提示有待审核信息，不能修改单位，单位调转信息填错怎么办，怎么删除人员，离职的人员怎么办，怎么调到临时单位",
    search_kwargs={"k": 10},
    chunk_size=100,
    separators=["\n\n"],
)

employing_unit_operation_tool = create_retrieval_tool(
    "./policies/operation/employing_unit_operation.md",
    "employing_unit_operation_engine",
    "回答用人单位学时申报、注册审核、信息变更、更换管理员、人员信息查询的系统操作相关问题，返回最相关的文档，如：单位怎么审核，怎么把人员调出单位，人员离职了怎么调出去，如何审核人员提交的学时，学时申报错了，单位也审核了怎么办，怎么驳回，学时申报错了，单位也审核了怎么办，单位培训计划，怎么提交、审核，怎么更换单位超级管理员，单位如何增加管理员，如何查询单位名下专技人员信息",
    search_kwargs={"k": 10},
    chunk_size=100,
    separators=["\n\n"],
)

supervisory_department_operation_tool = create_retrieval_tool(
    "./policies/operation/supervisory_department_operation.md",
    "supervisory_department_operation_engine",
    "回答主管部门注册审核、信息变更、继续教育机构审核、单位调转审核、学时申报审核、人员信息查询的系统操作相关问题，返回最相关的文档，如：如何审核单位或个人注册信息、人员或用人单位信息变更审核、如何审核继教机构信息、人员调入和单位调转审核操作、如何审核专技人员的学时、学时报错了，怎么驳回、学时申报错了，也审核通过了，还能驳回吗、如何查询主管部门下面单位情况",
    search_kwargs={"k": 10},
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

# online learning and test
online_learning_and_tests_tool = create_retrieval_tool(
    "./policies/online_learning_and_tests/online_learning_and_tests.md",
    "online_learning_and_tests_engine",
    "回答用户关于在线学习和考试的相关问题，返回最相关的文档，如：如何报班、怎么报名学习、公需课怎么报名、专业课怎么报名，济宁市高级职业学校/山东理工职业学院/微山县人民医院怎么报名课程、怎么补学、学习标准、年度学习要求是什么、学习到什么时候、什么时间能学、明年学行吗、课程没学完怎么办、用考试吗、必须考试吗、考试多少分合格、考试分数线是多少、考试有几次机会、我的考试在哪、怎么看考试",
    search_kwargs={"k": 8},
    chunk_size=100,
    separators=["\n\n"],
)

payments_tool = create_retrieval_tool(
    "./policies/payments/payments.md",
    "payments_engine",
    "回答用户关于支付的相关问题，返回最相关的文档，如：发票怎么开、能不能重开发票、发票错了能重开吗、发票列表在哪、怎么找发票、课程是怎么收费的、1学时多少钱、课程什么价格、课程报名有优惠吗、能便宜吗、集体缴费审核、集体缴费怎么退款、集体缴费用错卡支付",
    search_kwargs={"k": 8},
    chunk_size=100,
    separators=["\n\n"],
)

certificate_and_hours_tool = create_retrieval_tool(
    "./policies/certificate_and_hours/certificate_and_hours.md",
    "certificate_and_hours_engine",
    "回答用户关于证书和学时的相关问题，返回最相关的文档，如：怎么下载证书、怎么打印证书、证书打印、没有证书、为什么打印不了证书，公需课达标是多少、专业课达标是多少、达标要求、达标是什么标准，学时对接到哪、会对接到会计平台吗、会对接到济南市/德州市/东营市平台吗，会计网学的学时可以对接到省平台吗、在文旅厅平台学习的，学时没对接",
    search_kwargs={"k": 3},
    chunk_size=100,
    separators=["\n\n"],
)
    

# Create Agent
# model = Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3})
# model.model_name = "qwen-max"
# model.model_kwargs = {"temperature": 0.3}


tools = [
    # multiply,
    RegistrationStatusTool(),
    # AskForUserRoleTool(),
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
    individual_operation_tool,
    employing_unit_operation_tool,
    supervisory_department_operation_tool,
    personal_modify_info_tool,
    employ_supervise_modify_info_tool,
    cont_edu_modify_info_tool,
    complaints_tool,
    policy_inquiry_tool,
    other_questions_tool,
    online_learning_and_tests_tool,
    payments_tool,
    certificate_and_hours_tool
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
agent = create_react_agent(
    Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}), tools, prompt
)
main_qa_agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True
)


# create a router chain


# check user role agent
check_user_role_router_prompt = hub.pull("hwchase17/react")
check_user_role_router_prompt.template = """Your ONLY job is to determine the user role. DO NOT Answer the question.

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
check_user_role_router_prompt.input_variables = [
    "agent_scratchpad",
    "input",
    "tool_names",
    "tools",
]

check_user_role_router_tools = [CheckUserRoleTool()]

check_user_role_router_chain = create_react_agent(
    Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
    # ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.3),
    check_user_role_router_tools,
    check_user_role_router_prompt,
)
check_user_role_router_chain_executor = AgentExecutor.from_agent_and_tools(
    agent=check_user_role_router_chain,
    tools=check_user_role_router_tools,
    verbose=True,
    handle_parsing_errors=True,
)

# Update user role agent
update_user_role_prompt = hub.pull("hwchase17/react")
update_user_role_prompt.template = """Your ONLY job is to ask the user to provide their role information regardless of the input.

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
update_user_role_prompt.input_variables = [
    "agent_scratchpad",
    "input",
    "tool_names",
    "tools",
]

update_user_role_tools = [UpdateUserRoleTool()]

update_user_role_chain = create_react_agent(
    Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
    # ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.3),
    update_user_role_tools,
    update_user_role_prompt,
)
update_user_role_chain_executor = AgentExecutor.from_agent_and_tools(
    agent=update_user_role_chain,
    tools=update_user_role_tools,
    verbose=True,
    handle_parsing_errors=True,
)


# routing
def check_user_role_and_route(info):
    print(info["topic"])
    if "unknown" in info["topic"]["output"].lower():
        return update_user_role_chain_executor
    return main_qa_agent_executor


main_qa_chain = {
    "topic": check_user_role_router_chain_executor,
    "input": lambda x: x["input"],
} | RunnableLambda(check_user_role_and_route)


# check_is_credit_record_chain = check_is_credit_record_prompt | Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}) | StrOutputParser()

# this will be replaced with an agent
# general_chain = PromptTemplate.from_template("""Respond to the following user input:

# # Question: {input}
# # Answer:""") | Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3})

credit_problem_prompt = PromptTemplate.from_template(
    """Use a tool to answer the user's qustion.

You MUST use a tool and generate a response based on tool's output.
When user input a number longer than 6 digits, use it as user id number in the context for the tool.
When the user input a four-digit number, use it as year in the context for the tool.
DO NOT hallucinate!!!! DO NOT Assume any user inputs. ALWAYS ask the user for more information if needed.

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

# check user location
check_user_loc_router_prompt = hub.pull("hwchase17/react")
check_user_loc_router_prompt.template = """Your ONLY job is to determine the user location. DO NOT Answer the question.

NO MATTER WHAT, use a tool to find out the user location.
ALWAYS use a tool to check the user location.
You MUST use a tool to find out the user location.
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
check_user_loc_router_prompt.input_variables = [
    "agent_scratchpad",
    "input",
    "tool_names",
    "tools",
]

check_user_loc_router_tools = [CheckUserLocTool()]

check_user_loc_router_chain = create_react_agent(
    Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
    # ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.3),
    check_user_loc_router_tools,
    check_user_loc_router_prompt,
)
check_user_loc_router_chain_executor = AgentExecutor.from_agent_and_tools(
    agent=check_user_loc_router_chain,
    tools=check_user_loc_router_tools,
    verbose=True,
    handle_parsing_errors=True,
)

# update user location agent
update_user_location_prompt = hub.pull("hwchase17/react")
update_user_location_prompt.template = (
    """Your ONLY job is to ask the user to provide their location information regardless of the input.

You MUST ALWAYS say: 请问您是在哪个地市平台学习的？请先确认您的学习地市，以便我能为您提供相应的信息。我方负责的主要平台地市有：\n\n"""
    + LOC_STR
    + """

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
)
update_user_location_prompt.input_variables = [
    "agent_scratchpad",
    "input",
    "tool_names",
    "tools",
]

update_user_location_tools = [UpdateUserLocTool()]

update_user_location_chain = create_react_agent(
    Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
    # ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.3),
    update_user_location_tools,
    update_user_location_prompt,
)
update_user_location_chain_executor = AgentExecutor.from_agent_and_tools(
    agent=update_user_location_chain,
    tools=update_user_location_tools,
    verbose=True,
    handle_parsing_errors=True,
)


def check_user_loc_and_route(info):
    print(info["topic"])
    if "unknown" in info["topic"]["output"].lower():
        return update_user_location_chain_executor
    return credit_problem_chain_executor


main_credit_problem_chain = {
    "topic": check_user_loc_router_chain_executor,
    "input": lambda x: x["input"],
} | RunnableLambda(check_user_loc_and_route)

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


# how to register class
register_class_prompt = PromptTemplate.from_template(
    """Answer the user's question step by step. Don't give the whole answer at once. Guide the user to the solution.

Always start with Step 1 below, DO NOT go to Step 2. Only execute Step 1 first. Do Not include the keyword `Step 1` or `Step 2` in your response.

Step 1. First kindly ask the user whether they want to register 公需课 or 专业课

Step 2. Based on the user's choice in Step 1,
If the user wants 公需课, then say 选择【济宁职业技术学院】这个平台，进入【选课中心】，先选择【培训年度】，再选择对应年度的课程报名学习就可以。如果有考试，需要考试通过后才能计入对应年度的学时。
If the user wants 专业课, say 选择【济宁职业技术学院】这个平台，进入【选课中心】，先选择【培训年度】，再选择与您职称专业相符或者相关的课程进行报名，缴费后可以学习。专业课学完就可以计入对应年度的学时，无需考试。
If the user wants both, then say 如果要报名公需课，选择【济宁职业技术学院】这个平台，进入【选课中心】，先选择【培训年度】，再选择对应年度的课程报名学习就可以。如果有考试，需要考试通过后才能计入对应年度的学时。如果要报名专业课，选择【济宁职业技术学院】这个平台，进入【选课中心】，先选择【培训年度】，再选择与您职称专业相符或者相关的课程进行报名，缴费后可以学习。专业课学完就可以计入对应年度的学时，无需考试。
{chat_history}
Question: {input}
"""
)
register_class_prompt.input_variables = [
    "input",
    "chat_history",
]
register_class_llm = Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3})
register_class_memory = ConversationBufferMemory(
    memory_key="chat_history", input_key="input", return_messages=True
)
register_class_llm_chain = LLMChain(
    llm=register_class_llm,
    memory=register_class_memory,
    prompt=register_class_prompt,
    verbose=True,
    output_key="output",
)

# refund agent
refund_prompt = PromptTemplate.from_template(
    """Use a tool to answer the user's qustion.

Ask the user to provide 身份证号，in order to 查询课程信息
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

# refund_course_not_started_prompt = PromptTemplate.from_template(
#     """Now that user has confirmed that they have not started the course, you should say the following:
# 对于没有学习的课程，您可以点击右上方【我的学习】，选择【我的订单】，找到对应课程点击【申请售后】，费用在1个工作日会原路退回。

# DO NOT answer the question. Simply provide the above information.
                                                                
# Question: {input}

# Answer:"""
# )
# refund_course_not_started_prompt.input_variables = ["input"]
# refund_course_not_started_llm_chain = LLMChain(
#     llm=Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
#     prompt=refund_course_not_started_prompt,
#     verbose=True,
#     output_key="output",
# )

# refund_ask_if_started_prompt = PromptTemplate.from_template(
#     """You should say the following:
# 您要退费的课程学习了吗？

# DO NOT answer the question. Simply provide the above information.
                                                                
# Question: {input}

# Answer:"""
# )
# refund_ask_if_started_prompt.input_variables = ["input"]
# refund_ask_if_started_llm_chain = LLMChain(
#     llm=Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
#     prompt=refund_ask_if_started_prompt,
#     verbose=True,
#     output_key="output",
# )

# refund_router_prompt = PromptTemplate.from_template(
#     """Based on the chat history only, classify whether the user `学习了课程` or `没有学习课程` or `不知道学没学课程` or `用户未提供信息`.

# # Do not answer the question. Simply classify it as being related to `学习了课程` or `没有学习课程` or `不知道学没学课程` or `用户未提供信息`.
# # Do not respond with anything other than `学习了课程` or `没有学习课程` or `不知道学没学课程` or `用户未提供信息`.

# {chat_history}
# Question: {input}

# # Classification:"""
# )
# refund_router_prompt.input_variables = ["input", "chat_history"]
# refund_router_memory = ConversationBufferMemory(
#     memory_key="chat_history", input_key="input"
# )
# refund_router_llm_chain = LLMChain(
#     llm=Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
#     prompt=refund_router_prompt,
#     memory=refund_router_memory,
#     verbose=True,
# )


# def refund_route(info):
#     print(info)
#     if "学习了课程" in info["topic"]["text"]:
#         print("学习了课程")
#         return refund_chain_executor
#     if "没有学习课程" in info["topic"]["text"]:
#         print("没有学习课程")
#         return refund_course_not_started_llm_chain
#     if "不知道学没学课程" in info["topic"]["text"]:
#         print("不知道学没学课程")
#         return refund_chain_executor
#     if "用户未提供信息" in info["topic"]["text"]:
#         print("用户未提供信息")
#         return refund_ask_if_started_llm_chain


# refund_full_chain = {
#     "topic": refund_router_llm_chain,
#     "input": lambda x: x["input"],
# } | RunnableLambda(refund_route)

# cannot find course agent
cannot_find_course_prompt = PromptTemplate.from_template(
    """Use a tool to answer the user's qustion.

You MUST use a tool and generate a response based on tool's output.

When user input a number longer than 6 digits, use it as user id number in the context for the tool.
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
# check if the question topic
# check_is_credit_record_prompt = PromptTemplate.from_template("""Given the user input AND chat history below, classify it as either being about `学时没显示, 学时有问题` or `other`

# # Do not answer the question. Simply classify it as being related to `学时没显示` or `学时有问题` or `学时申报` or `学时审核` or `other`.
# # Do not respond with anything other than `学时没显示` or `学时有问题` or `学时申报` or `学时审核` or `other`.

# {chat_history}
# Question: {input}

# # Classification:""")

template = """Given the user input AND chat history below, classify whether the user's topic being about `学时没显示` or `学时有问题` or `学时申报` or `学时审核` or `课程进度` or `多个设备，其他地方登录` or `课程退款退费，课程买错了` or `课程找不到，课程没有了` or `other`.

# Do not answer the question. Simply classify it as being related to `学时没显示` or `学时有问题` or `学时申报` or `学时审核` or `课程进度` or `多个设备，其他地方登录` or `课程退款退费，课程买错了` or `课程找不到，课程没有了` or `other`.
# Do not respond with anything other than `学时没显示` or `学时有问题` or `学时申报` or `学时审核` or `课程进度` or `多个设备，其他地方登录` or `课程退款退费，课程买错了` or `课程找不到，课程没有了` or `other`.

{chat_history}
Question: {input}

# Classification:"""

check_is_credit_record_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template=template,
)

check_is_credit_record_memory = ConversationBufferMemory(
    memory_key="chat_history", input_key="input"
)

check_is_credit_record_chain = LLMChain(
    llm=Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
    prompt=check_is_credit_record_prompt,
    memory=check_is_credit_record_memory,
    verbose=True,
)


def check_is_credit_record_router(info):
    print(info)
    if "学时没显示" in info["topic"]["text"]:
        print("学时没显示")
        return main_credit_problem_chain
    if "学时有问题" in info["topic"]["text"]:
        print("学时有问题")
        return main_credit_problem_chain
    if "学时申报" in info["topic"]["text"]:
        print("学时申报")
        return main_qa_chain
    if "学时审核" in info["topic"]["text"]:
        print("学时审核")
        return main_qa_chain
    if "other" in info["topic"]["text"]:
        print("other")
        return main_qa_chain
    if "课程进度" in info["topic"]["text"]:
        print("课程进度")
        return course_progress_problems_llm_chain
    if "多个设备，其他地方登录" in info["topic"]["text"]:
        print("多个设备，其他地方登录")
        return multiple_login_llm_chain
    if "课程退款退费，课程买错了" in info["topic"]["text"]:
        print("课程退款退费，课程买错了")
        return refund_chain_executor
        # return refund_full_chain
    if "课程找不到，课程没有了" in info["topic"]["text"]:
        print("课程找不到，课程没有了")
        return cannot_find_course_chain_executor
    print("unknown")
    return main_qa_chain


full_chain = {
    "topic": check_is_credit_record_chain,
    "input": lambda x: x["input"],
} | RunnableLambda(check_is_credit_record_router)


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
            print(response)
            # st.write(response)
            st.write(response["output"])
            message = {"role": "assistant", "content": response["output"]}
            # message = {"role": "assistant", "content": response}
            # st.session_state.chat_engine.memory.add_message(message)
            st.session_state.messages.append(message)  # Add response to message history
