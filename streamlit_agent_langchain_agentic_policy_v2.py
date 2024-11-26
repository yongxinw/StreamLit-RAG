import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import LLMChain
from langchain.tools.render import render_text_description

from langchain.embeddings.dashscope import DashScopeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Tongyi
from langchain_community.vectorstores import FAISS
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
    DASHSCOPE_API_KEY,
    SIM_DATA,
    LLM_NAME,
)
from utils import (
    check_user_location,
    create_atomic_retriever_agent,
    create_atomic_retriever_agent_single_tool_qa_map,
    create_dummy_agent,
    create_react_agent_with_memory,
    create_single_function_call_agent,
    output_parser,
    create_retrieval_tool,
)
from tools import *
from prompts import *
import pdb


def _init():
    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
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
                "content": f"""欢迎您来到大众云学，我是大众云学的专家助手，我可以回答关于大众云学的所有问题。
                测试请使用身份证号372323199509260348。测试公需课/专业课学时，请使用年份2019/2020。
                测试课程购买，退款等，请使用年份2023，课程名称新闻专业课培训班。测试模拟数据如下：\n\n
                {SIM_DATA}
        """,
            }  # TODO: ask user type here?
        ]


_init()


# Dynamically updated tools stay in this file.
# CheckUserCreditTool, UpdateUserLocTool2, UpdateUserRoleTool2
class CheckUserCreditTool(BaseTool):
    """根据用户回答，检查用户学时状态"""

    name: str = "检查用户学时状态工具"
    description: str = (
        "用于检查用户学时状态，需要指通过 json 指定用户身份证号 user_id_number、用户想要查询的年份 year、用户想要查询的课程类型 course_type "
    )
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

        return apis.check_credit_hours_api(params_dict, credit_problem_chain_executor)


class UpdateUserLocTool2(BaseTool):
    """根据用户回答，更新用户学习地市"""

    name: str = "用户学习地市更新工具"
    description: str = (
        "用于更新用户学习地市，需要指通过 json 指定用户学习地市 user_location "
    )
    return_direct: bool = True

    def _run(self, params) -> Any:
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

        if (
            params_dict is None
            or "user_location" not in params_dict
            or params_dict["user_location"] is None
            or params_dict["user_location"] == "unknown"
            or (
                params_dict["user_location"] not in LOC_STR
                and params_dict["user_location"]
                not in ["开放大学", "蟹壳云学", "专技知到", "文旅厅", "教师"]
            )
        ):
            return (
                "请问您是在哪个地市平台学习的？请先确认您的学习地市，以便我能为您提供相应的信息。我方负责的主要平台地市有：\n\n"
                + LOC_STR
            )
        user_location = params_dict["user_location"]
        credit_problem_chain_executor.agent.runnable.get_prompts()[0].template.replace(
            "user location: unknown", f"user location: {user_location}"
        )
        return f"谢谢，已为您更新您的学习地市为{user_location}, 现在请您提供身份证号码，以便我查询您的学时状态。"


# ===========================================================================
#  START: MainChain - check user role
# ===========================================================================
class UpdateUserRoleTool2(BaseTool):
    """根据用户回答，更新用户角色"""

    name: str = "用户角色更新工具"
    description: str = (
        "用于更新用户在对话中的角色，需要指通过 json 指定用户角色 user_role "
    )
    return_direct: bool = True

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
        if (
            params_dict is None
            or "user_role" not in params_dict
            or params_dict["user_role"] is None
            or params_dict["user_role"]
            not in ["专技个人", "用人单位", "主管部门", "继续教育机构", "跳过"]
        ):
            return '您好，目前我们支持的用户类型为专技个人，用人单位，主管部门和继续教育机构，请问您想咨询那个用户类型？（回复"跳过"默认进入专技个人用户类型）'

        user_role = params_dict["user_role"]
        if user_role == "跳过":
            user_role = "专技个人"
        main_qa_agent_executor.agent.runnable.get_prompts()[0].template.replace(
            "Current user role is unknown", f"Current user role is: {user_role}"
        )
        return f"更新您的用户角色为{user_role}, 请问有什么可以帮到您？"


prompt = PromptTemplate.from_template(USER_ROLE_PROMPT)
prompt.input_variables = [
    "agent_scratchpad",
    "input",
    "tool_names",
    "tools",
    "chat_history",
]

tools = [
    RegistrationStatusTool(),
    UpdateUserRoleTool2(),
]

memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")
agent = create_react_agent(
    Tongyi(model_name=LLM_NAME, model_kwargs={"temperature": 0.3}), tools, prompt
)
main_qa_agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True
)
# pdb.set_trace()


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
#  END: MainChain - Check user role
# ===========================================================================


# ===========================================================================
#  START: MainChain - Check user router
# ===========================================================================
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
#  END: MainChain - Check user router
# ===========================================================================


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


login_problem_classifier_prompt = PromptTemplate.from_template(LOGIN_PROMPT)

login_problem_classifier_prompt.input_variables = ["input", "chat_history"]

login_problem_classifier_memory = ConversationBufferMemory(
    memory_key="chat_history", input_key="input"
)

login_problem_classifier_chain = LLMChain(
    llm=Tongyi(model_name=LLM_NAME, model_kwargs={"temperature": 0.3}),
    prompt=login_problem_classifier_prompt,
    memory=login_problem_classifier_memory,
    verbose=True,
)

login_problem_agent_executor = create_atomic_retriever_agent_single_tool_qa_map(
    login_problems_detail_tool,
    qa_map_path="./policies_v2/login_problems_details_qa_map.json",
)

login_problem_ask_user_executor = LLMChain(
    llm=Tongyi(model_name=LLM_NAME, model_kwargs={"temperature": 0.3}),
    prompt=PromptTemplate.from_template(LOGIN_ASK_PROMPT),
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
forgot_password_tool = create_retrieval_tool(
    "./policies_v2/forgot_password_q.md",
    "forgot_password_engine",
    "回答用户忘记密码的相关问题，返回最相关的文档，如：忘记密码怎么办，密码忘记了，找回密码，忘记密码手机号那里怎么是空的、手机号不显示、手机号怎么修改、手机号不用了，怎么找回、姓名或身份证号或所在单位有误、提示什么姓名错误、身份证号错误、所在单位有误、密码怎么保存不了、改密码怎么不行、改密码怎么保存不了、密码保存不了",
    search_kwargs={"k": 3},
    chunk_size=100,
    separators=["\n\n"],
)

forgot_password_classifier_prompt = PromptTemplate.from_template(FORGOT_PASSWORD_PROMPT)

forgot_password_classifier_prompt.input_variables = ["input", "chat_history"]
forgot_password_classifier_memory = ConversationBufferMemory(
    memory_key="chat_history", input_key="input"
)
forgot_password_classifier_chain = LLMChain(
    llm=Tongyi(model_name=LLM_NAME, model_kwargs={"temperature": 0.3}),
    prompt=forgot_password_classifier_prompt,
    memory=forgot_password_classifier_memory,
    verbose=True,
)

forgot_password_agent_executor = create_atomic_retriever_agent_single_tool_qa_map(
    forgot_password_tool, qa_map_path="./policies_v2/forgot_password_qa_map.json"
)

forgot_password_ask_user_executor = LLMChain(
    llm=Tongyi(model_name=LLM_NAME, model_kwargs={"temperature": 0.3}),
    prompt=PromptTemplate.from_template(FORGOT_PASSWORD_ASK_PROMPT),
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
#  START: CREDIT CHAIN
# ===========================================================================
credit_problem_prompt = PromptTemplate.from_template(CREDIT_PROBLEM_PROMPT)
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
    Tongyi(model_name=LLM_NAME, model_kwargs={"temperature": 0.3}),
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
#  END: CREDIT CHAIN
# ===========================================================================


# ===========================================================================
#  START: Courses Progress
# ===========================================================================
course_progress_problems_prompt = PromptTemplate.from_template(COURSE_PROGRESS_PROMPT)
course_progress_problems_prompt.input_variables = [
    "input",
    "chat_history",
]
course_progress_problems_llm = Tongyi(
    model_name=LLM_NAME, model_kwargs={"temperature": 0.3}
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
multiple_login_prompt = PromptTemplate.from_template(MULTIPLE_LOGIN_PROMPT)
multiple_login_prompt.input_variables = [
    "input",
    "chat_history",
]
multiple_login_llm = Tongyi(model_name=LLM_NAME, model_kwargs={"temperature": 0.3})
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
refund_prompt = PromptTemplate.from_template(REFUND_PROMPT)

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
    Tongyi(model_name=LLM_NAME, model_kwargs={"temperature": 0.3}),
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
cannot_find_course_prompt = PromptTemplate.from_template(CANT_FIND_COURSE_PROMPT)
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
    Tongyi(model_name=LLM_NAME, model_kwargs={"temperature": 0.3}),
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


# ********************************************************************************
# MAIN ENTRY POINT
# ********************************************************************************

main_question_classifier_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template=MAIN_QUESTION_CLASSIFIER_PROMPT,
)

main_question_classifier_mem = ConversationBufferMemory(
    memory_key="chat_history", input_key="input"
)

main_question_classifier = LLMChain(
    llm=Tongyi(model_name=LLM_NAME, model_kwargs={"temperature": 0.3}),
    prompt=main_question_classifier_prompt,
    memory=main_question_classifier_mem,
    verbose=True,
)

if "topic" not in st.session_state:
    st.session_state.topic = None

# Main functional chain (not main chain).
qa_chain_v2 = {
    "topic": check_user_role_chain,
    "input": lambda x: x["input"],
} | RunnableLambda(check_role_qa_router)


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
        return jining_agent_executor

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


# Main chain.
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
