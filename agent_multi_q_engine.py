import openai
import streamlit as st
from llama_index import (
    Document,
    ServiceContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI
from llama_index.tools import BaseTool, FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.prompts import PromptTemplate
from statics import REGISTRATION_STATUS

agent_template_str = (
    '\nYou are designed to help with a variety of tasks, from answering questions     to providing summaries to other types of analyses. \n\n'
    '## Tools\nYou have access to a wide variety of tools. You are responsible for using\n'
    'the tools in any sequence you deem appropriate to complete the task at hand.\n'
    'This may require breaking the task into subtasks and using different tools\n'
    'to complete each subtask.\n\n'
    'You have access to the following tools:\n{tool_desc}\n\n## Output Format\n'
    'To answer the question, please use the following format.\n\n'
    '```\nThought: I need to use a tool to help me answer the question.\n'
    'Action: tool name (one of {tool_names}) if using a tool.\n'
    'Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})\n'
    '```\n\nPlease ALWAYS start with a Thought.\n\n'
    'Please use a valid JSON format for the Action Input. Do NOT do this {{\'input\': \'hello world\', \'num_beams\': 5}}.\n\n'
    'If this format is used, the user will respond in the following format:\n\n'
    '```\nObservation: tool response\n```\n\n'
    'You should keep repeating the above format until you have enough information\n'
    'to answer the question without using any more tools. At that point, you MUST respond\n'
    'in the one of the following two formats:\n\n```\n'
    'Thought: I can answer without using any more tools.\n'
    'Answer: [your answer here]\n```\n\n```\n'
    'Thought: I cannot answer the question with the provided tools.\n'
    'Answer: Sorry, I cannot answer your query.\n```\n\n'
    'ALWAYS check user role from chat history before any actions.\n'
    'When user role is unknown, you MUST use the registration_engine to fetch available roles and ask the user for his role based on registration_engine output\n'
    'You MUST NOT use any tools to infer user role or ask directly.\n'
    'When user has provided role information, use the correct tool to update user role and proceed with the answering questions.\n'
    'Current user role is unknown\n\n'
    'All conversation is in Chinese. Please use Chinese for all conversation.\n\n'
    '## Current Conversation\n'
    'Below is the current conversation consisting of interleaving human and assistant messages.\n\n'
)


def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}\n" f"**Text:** \n"
        print(text_md)
        print(p.get_template())


openai.api_key = st.secrets.openai_key

llm = OpenAI(
    model="gpt-4-turbo-preview",
    # model="gpt-3.5-turbo-0613",
    # model="gpt-4-0613",
    # model="gpt-4-0125-preview",
    temperature=0.4,
    system_prompt="你是一个关于大众云学的专家，你了解关于大众云学的所有问题。用户角色未知时，请先询问角色。假设所有的问题都与大众云学有关。保持你的答案技术性和基于事实——不要产生幻觉。",
)


def load_data(input_dir=None, input_files=None, recursive=True):
    reader = SimpleDirectoryReader(input_dir=input_dir, input_files=input_files, recursive=True)
    docs = reader.load_data()
    service_context = ServiceContext.from_defaults(llm=llm)
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return index


def multiply(a: int, b: int) -> int:
    """计算两个数的乘积并返回结果"""
    return a * b


def update_user_role(input: str = "123"):
    """
    根据语境，更新用户角色以及对应的回答模板，以便更好地回答用户问题。
    """
    # TODO: This is a hacky fix. A better solution: https://stackoverflow.com/questions/11283961/partial-string-formatting
    USER_ROLE = input
    policy_engine_tmpl_str = (
        "注意：回答问题前，请先从用户对话中尝试确定用户角色，若无法推测，则询问用户角色，不要推测用户角色，回答问题时请保持技术性和基于事实，不要产生幻觉。\n"
        #  "注意：若用户角色为未知，请先询问用户角色，不要推测用户角色，回答问题时请保持技术性和基于事实，不要产生幻觉。\n"
        "注意：用户角色为{user_role}\n"
        "语境信息如下\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "请根据语境信息，不要使用先验知识，回答下面的问题。\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    print(USER_ROLE)
    template_str = policy_engine_tmpl_str.format(
        context_str="{context_str}", user_role=USER_ROLE if USER_ROLE is not None else "未知", query_str="{query_str}"
    )
    print(template_str)
    policy_engine_tmpl = PromptTemplate(template_str)
    agent.agent_worker._get_tools("")[0]._query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": policy_engine_tmpl}
    )

    agent_template_str = (
        '\nYou are designed to help with a variety of tasks, from answering questions     to providing summaries to other types of analyses. \n\n'
        '## Tools\nYou have access to a wide variety of tools. You are responsible for using\n'
        'the tools in any sequence you deem appropriate to complete the task at hand.\n'
        'This may require breaking the task into subtasks and using different tools\n'
        'to complete each subtask.\n\n'
        'You have access to the following tools:\n{tool_desc}\n\n## Output Format\n'
        'To answer the question, please use the following format.\n\n'
        '```\nThought: I need to use a tool to help me answer the question.\n'
        'Action: tool name (one of {tool_names}) if using a tool.\n'
        'Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})\n'
        '```\n\nPlease ALWAYS start with a Thought.\n\n'
        'Please use a valid JSON format for the Action Input. Do NOT do this {{\'input\': \'hello world\', \'num_beams\': 5}}.\n\n'
        'If this format is used, the user will respond in the following format:\n\n'
        '```\nObservation: tool response\n```\n\n'
        'You should keep repeating the above format until you have enough information\n'
        'to answer the question without using any more tools. At that point, you MUST respond\n'
        'in the one of the following two formats:\n\n```\n'
        'Thought: I can answer without using any more tools.\n'
        'Answer: [your answer here]\n```\n\n```\n'
        'Thought: I cannot answer the question with the provided tools.\n'
        'Answer: Sorry, I cannot answer your query.\n```\n\n'
        'ALWAYS check user role from chat history before any actions.\n'
        'When user role is unknown, you MUST use the registration_engine to fetch available roles and ask the user for his role based on registration_engine output\n'
        'You MUST NOT use any tools to infer user role or ask directly.\n'
        'When user has provided role information, use the correct tool to update user role and proceed with the answering questions.\n'
        'Current user role is' + USER_ROLE + '\n\n'
        'If the user is unsure of whether they have registered, you MUST ask them to provide the administrator ID number and THEN use the right tool to check the registration status.\n\n'
        '## IMPORTANT: \n'
        'All conversation is in Chinese. Please use Chinese for all conversation.\n\n'
        '## Current Conversation\n'
        'Below is the current conversation consisting of interleaving human and assistant messages.\n\n'
    )
    agent.update_prompts({"agent_worker:system_prompt": PromptTemplate(agent_template_str)})
    return "user role updated"

def lookup_by_id(input: str = "123"):
    """
    根据用户输入的ID号，查询用户在大众云学平台上的注册状态。
    """
    if REGISTRATION_STATUS.get(input) is not None:
        return REGISTRATION_STATUS.get(input)
    return "经查询，您尚未在大众云学平台上注册"

multiply_tool = FunctionTool.from_defaults(
    fn=multiply,
    tool_metadata=ToolMetadata(name="multiply", description="计算两个数的乘积并返回结果。"),
)

update_user_role_tool = FunctionTool.from_defaults(
    fn=update_user_role,
    tool_metadata=ToolMetadata(
        name="update_role", description="根据语境，更新用户角色以及对应的回答模板，以便更好地回答用户问题。"
    ),
)

lookup_by_id_tool = FunctionTool.from_defaults(
    fn=lookup_by_id,
    tool_metadata=ToolMetadata(
        name="lookup_by_id",
        description="根据用户输入的ID号，查询用户在大众云学平台上的注册状态。",
    ),
)

def get_query_engine_tool(input_dir, input_files, tool_name, description):
    policy_engine_tmpl_str = (
        "注意：回答问题前，请先从用户对话中尝试确定用户角色，若无法推测，则询问用户角色，不要推测用户角色，回答问题时请保持技术性和基于事实，不要产生幻觉。\n"
        # "注意：若用户角色为未知，请先询问用户角色，不要推测用户角色，回答问题时请保持技术性和基于事实，不要产生幻觉。\n"
        "注意：用户角色为未知\n"
        "语境信息如下\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "请根据语境信息，不要使用先验知识，回答下面的问题。\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    policy_engine_refine_tmpl_str = (
        "以下是原始查询：{query_str}\n"
        "我们已经提供了一个现有的答案：{existing_answer}\n"
        "我们有机会通过下面的一些更多上下文来改进现有的答案（仅在需要时）。\n"
        "------------\n"
        "{context_msg}"
        "------------\n"
        "根据新的上下文，改进原始答案以更好地回答查询。如果上下文没有用，返回原始答案。\n"
        "Refined Answer:"
    )
    index = load_data(input_dir, input_files)
    policy_engine = index.as_query_engine(
        similarity_top_k=5,
        verbose=True,
    )
    policy_engine_tmpl = PromptTemplate(policy_engine_tmpl_str)
    policy_enging_refine_tmpl = PromptTemplate(policy_engine_refine_tmpl_str)
    policy_engine.update_prompts(
        {"response_synthesizer:text_qa_template": policy_engine_tmpl}
    )
    policy_engine.update_prompts(
        {"response_synthesizer:refine_template": policy_enging_refine_tmpl}
    )
    policy_query_tool = QueryEngineTool(
        query_engine=policy_engine,
        metadata=ToolMetadata(
            name=tool_name,
            description=description,
        ),
    )
    return policy_query_tool

tools = [
    # policy_query_tool,
    # multiply_tool,
    get_query_engine_tool(
        input_dir=None,
        input_files=["./policies/registration/registration.md"],
        tool_name="registration_engine",
        description="负责查询大众云学平台的注册方法，返回最相关的文档",
    ),
    get_query_engine_tool(
        input_dir=None,
        input_files=["./policies/registration/auditing.md"],
        tool_name="auditing_engine",
        description="负责回答关于注册审核的相关问题，返回最相关的文档",
    ),
    get_query_engine_tool(
        input_dir=None,
        input_files=["./policies/registration/withdrawal_and_modification.md"],
        tool_name="withdrawal_engine",
        description="负责回答关于如何撤回、修改、驳回注册的问题，返回最相关的文档",
    ),
    get_query_engine_tool(
        input_dir=None,
        input_files=["./policies/registration/professional_individual_reg_page_faq.md"],
        tool_name="professional_individual_registration_faq_engine",
        description="负责回答专技个人注册页面细项，以及常见问题，返回最相关的文档",
    ),
    get_query_engine_tool(
        input_dir=None,
        input_files=["./policies/registration/employing_unit_reg_page_faq.md"],
        tool_name="employing_unit_registration_faq_engine",
        description="负责回答用人单位注册页面细项，以及常见问题，返回最相关的文档",
    ),
    get_query_engine_tool(
        input_dir=None,
        input_files=["./policies/registration/supervisory_department_reg_page_faq.md"],
        tool_name="supervisory_department_registration_faq_engine",
        description="负责回答主管部门注册页面细项，以及常见问题，返回最相关的文档",
    ),
    get_query_engine_tool(
        input_dir=None,
        input_files=["./policies/registration/continuing_edu_inst_reg_page_faq.md"],
        tool_name="continuing_education_institute_registration_faq_engine",
        description="负责回答继续教育机构注册页面细项，以及常见问题，返回最相关的文档",
    ),
    update_user_role_tool,
    lookup_by_id_tool
]

agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)
agent.update_prompts({"agent_worker:system_prompt": PromptTemplate(agent_template_str)})

import ipdb

ipdb.set_trace()
