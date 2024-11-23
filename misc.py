# ================================================================================
# START: Check user role
# ================================================================================


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


# check user role agent
check_user_role_router_prompt = PromptTemplate.from_template(
    """Your ONLY job is to determine the user role. DO NOT Answer the question.

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
)
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
# ================================================================================
# END: Check user role
# ================================================================================


# ================================================================================
# START: Update user role
# ================================================================================

# Update user role agent
update_user_role_prompt = PromptTemplate.from_template(
    """Your ONLY job is to ask the user to provide their role information regardless of the input.

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
)
update_user_role_prompt.input_variables = [
    "agent_scratchpad",
    "input",
    "tool_names",
    "tools",
]

update_user_role_tools = [UpdateUserRoleTool()]

update_user_role_chain = create_react_agent(
    Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
    update_user_role_tools,
    update_user_role_prompt,
)
update_user_role_chain_executor = AgentExecutor.from_agent_and_tools(
    agent=update_user_role_chain,
    tools=update_user_role_tools,
    verbose=True,
    handle_parsing_errors=True,
)
# ================================================================================
# END: Update user role
# ================================================================================


# ================================================================================
# START: Register class
# ================================================================================

# register_class_prompt = PromptTemplate.from_template(
#     """分步回答用户的问题。不要一次性给出所有答案。引导用户解决关于报班报课，以及费用的问题。

#     ## 报班报课及费用政策
#     ### 报班报课的信息如下：
#     公需课 -> 选择【济宁职业技术学院】这个平台，进入【选课中心】，先选择【培训年度】，再选择对应年度的课程报名学习就可以。如果有考试，需要考试通过后才能计入对应年度的学时。
#     专业课 -> 选择【济宁职业技术学院】这个平台，进入【选课中心】，先选择【培训年度】，再选择与您职称专业相符或者相关的课程进行报名，缴费后可以学习。专业课学完就可以计入对应年度的学时，无需考试。

#     ### 报班报课的费用如下：
#     公需课：经当地人社规定要求，公需课免费
#     专业课：经当地人社规定要求，专业课价格为1元1学时

#     ### 优惠
#     抱歉，课程价格是根据人社要求设定，没有优惠政策，需按照课程标定的价格购买。

#     ### 集体缴费
#     集体报班提交后需要人工审核，请耐心等待，及时关注审核情况。
#     集体缴费退费需要人工处理，请将支付截图、退款原因发送到我方邮箱，邮箱号为：sdzjkf@163.com，请及时关注邮箱回复并按照要求提供相关信息。
#     集体缴费换卡支付需要人工处理，请将您支付的带商户单号的截图、金额两个信息发送到我方邮箱，邮箱号为：sdzjkf@163.com，等待1-3个工作日后，您直接点击“换卡支付”按钮，微信立即支付即可。支付完成之后上一笔订单的费用自动退款。

#     ### 济宁市高级职业学校/山东理工职业学院/微山县人民医院怎么报名课程？
#     抱歉，我们只负责济宁职业技术学院这个培训平台，其他培训学校进入具体的培训平台进行咨询

#     ## 指南：
#     永远从下面的第1步开始，不要直接跳到第2步。在回答中不要包含关键字 `第1步` 或 `第2步`。

#     第1步. 首先询问用户是否要注册 公需课 or 专业课

#     第2步. 根据用户在第1步中的选择，在报班报课及费用政策中，选择最相关的回答。
#     如果用户想要 公需课, 则只回答公需课相关的信息
#     如果用户想要 专业课, 则只回答专业课相关的信息
#     如果用户都想了解，则将两个回答都提供。

#     ### 注意：
#     在询问用户是否要注册 公需课 or 专业课 前，不要直接回答用户的问题。始终引导用户解决问题。
#     请保持回答简洁，直接。始终使用中文回答问题

# {chat_history}
# 问题: {input}
# """
# )

# register_class_prompt.input_variables = [
#     "input",
#     "chat_history",
# ]
# register_class_llm = Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3})
# register_class_memory = ConversationBufferMemory(
#     memory_key="chat_history", input_key="input", return_messages=True
# )
# register_class_llm_chain = LLMChain(
#     llm=register_class_llm,
#     memory=register_class_memory,
#     prompt=register_class_prompt,
#     verbose=True,
#     output_key="output",
# )
# ================================================================================
# END: Register class
# ================================================================================


# ================================================================================
# START: Intent Classifier
# ================================================================================

# intent_classifier_template = """给定用户输入，判断用户的目的是否是提供用户角色信息，回答 `是` 或 `否`。

# 不要回答用户的问题。仅把用户的问题归类为 `是` 或 `否`。不要回答除此之外的任何内容。

# 用户角色为：专技个人、用人单位、主管部门、继续教育机构、跳过

# 注意：用户的问题可能包含角色，即使包含角色，用户的意图不一定是提供角色信息。因此，当包含角色时，你要更加小心的对用户的意图进行分类。

# # 以下是一些例子：

# 问题: {input}

# Classification:"""

# intent_classifier_prompt = PromptTemplate(
#     input_variables=["input", "chat_history"],
#     template=intent_classifier_template,
# )

# intent_classifier_mem = ConversationBufferMemory(
#     memory_key="chat_history", input_key="input"
# )

# intent_classifier = LLMChain(
#     llm=Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
#     prompt=intent_classifier_prompt,
#     memory=intent_classifier_mem,
#     verbose=True,
# )
# ================================================================================
# END: Intent Classifier
# ================================================================================
