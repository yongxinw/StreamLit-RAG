from langchain.agents import AgentExecutor, create_react_agent

from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Tongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from statics import (COURSE_PURCHASES, CREDIT_HOURS, LOC_STR,
                     REGISTRATION_STATUS)


def create_react_agent_with_memory(tools, prompt_str=None):
    if prompt_str is None:
        prompt_str = """Your ONLY job is to use a tool to answer the following question.

        You MUST use a tool to answer the question. 
        Simply Answer "您能提供更多关于这个问题的细节吗？" if you don't know the answer.
        DO NOT answer the question without using a tool.
        
        Please keep your answers short and to the point.

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
    else:
        prompt_str = prompt_str

    prompt = PromptTemplate.from_template(prompt_str)
    prompt.input_variables = [
        "agent_scratchpad",
        "input",
        "tool_names",
        "tools",
        "chat_history",
    ]

    memory = ConversationBufferMemory(
        memory_key="chat_history", input_key="input"
    )
    agent = create_react_agent(
        Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
        tools=tools,
        prompt=prompt,
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent_executor

def create_dummy_agent(dummy_message):
    dummy_agent = create_react_agent(
        Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
        tools={},
        prompt=PromptTemplate.from_template(dummy_message),
    )
    dummy_agent_executor = AgentExecutor.from_agent_and_tools(
        agent=dummy_agent,
        tools={},
        verbose=True,
        handle_parsing_errors=True,
    )
    return dummy_agent_executor

def create_atomic_retriever_agent(tools, summarization_llm_prompt, summarization_llm, system_prompt=None):

    # rendered_tools = render_text_description([individual_qa_tool, RegistrationStatusToolIndividual()])
    rendered_tools = render_text_description(tools)

    if system_prompt is None:
        system_prompt = f"""You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

        {rendered_tools}
        
        If you find none of the tools relevant, you can use the {tools[0].name} to answer the question.

        A few examples below:
        - user: "我想知道我的注册状态", 调用 {tools[1].name}
        - user: "292993194919231411", 调用 {tools[1].name}
        - user: "怎么查看注册待审核信息", 调用 {tools[0].name}
        - user: "怎么审核？", 调用 {tools[0].name}

        Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys. 'argument' value should be a json with the input to the tool."""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{input}")]
    )
    
    model = Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3})
    chain = prompt | model | JsonOutputParser()

    def tool_chain(model_output):
        tool_map = {tool.name: tool for tool in tools}
        print(model_output)
        chosen_tool = tool_map[model_output["tool_use"]["name"]]
        if "qa_engine" in chosen_tool.name:
            def _parse_retreiver_inputs(model_output):
                return model_output["question"]["input"]
            return {"context": _parse_retreiver_inputs | chosen_tool, "input": RunnablePassthrough()} | summarization_llm_prompt | summarization_llm | output_parser
        
        return {"params": RunnableLambda(lambda x: x["tool_use"]["arguments"])} | chosen_tool | output_parser

    runnable = RunnableParallel(
        question=RunnablePassthrough(),
        tool_use=chain
    )
    return runnable | tool_chain

def output_parser(model_output):
    return {"output": model_output}

def create_single_function_call_agent(tool, system_prompt=None):
    rendered_tools = render_text_description([tool])

    if system_prompt is None:
        system_prompt = f"""You are an assistant that has access to one and only one tool. Here is the name and description for it:

        {rendered_tools}

        If you think none of the tools are relevant, default to using {tool.name}.
        Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys. 'argument' value should be a json with the input to the tool."""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{input}")]
    )

    model = Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3})
    chain = prompt | model | JsonOutputParser()
    return chain | {"params": RunnableLambda(lambda x: x["arguments"])} | tool | output_parser