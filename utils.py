import os
import json
import operator
from langchain.agents import AgentExecutor, create_react_agent

from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Tongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain


from statics import (COURSE_PURCHASES, CREDIT_HOURS, LOC_STR,
                     REGISTRATION_STATUS)

os.environ["DASHSCOPE_API_KEY"] = "sk-91ee79b5f5cd4838a3f1747b4ff0e850"


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

#     """Answer the user's question based on the context provided below. The context contains the top 3 matches from the database separated by changing lines, the first one has the highest matching score.

# There are several rules:
# 1. You should only select relavent part in the context to answer the question.
# 2. Focus more on the top answer, you can drop the other answers if needed.
# 3. You don't need to use all the information, some provided information could be irrelavant.
# 4. Make the answer concise and clear.
# 5. Try not to change the content too much.
# 6. Don't hallucinate.

# 比如说，对于问题 "学时对接到哪"， 跟定的参考内容是:

# 学时对接到哪 ： 本平台的学时只会对接到职称平台，也就是 山东省专业技术人员管理服务平台，不会对接地市等其他平台，建议您以本平台的学时为准

# 怎么学时申报、学时申报怎么操作、如何学时申报 ： 登陆专技个人账号，进入【学时申报】，根据右侧页面的提示，填写相关信息，提交后，需要有单位审核通过后才可计入学时。

# 会计网学的学时可以对接到省平台吗 ：本平台只是接收方，学时如果和您实际不符，建议您先咨询您的学习培训平台，学时是否有正常推送过来，只有推送了我们才能收到，才会显示对应学时。

# 在这种情况下，你主要应该看第一个参考问题和回答，后面两个并不相关。
# 以下内容包括问题匹配度最高的三组答案，第一组答案往往是最相关的，有时候其他答案也会有帮助。
merge_results_prompt = PromptTemplate.from_template(
    """ 你的任务是根据以下内容，用中文总结并回答用户的问题 “{input}”。
    如果用户提供了反馈或建议，请从参考内容中提取最相关的回复话术，总结并回复。

    参考内容：{context}
    
    只需要在参考内容中选择相关的部分进行总结，可以舍弃不相关的部分。
    不要添加任何新的信息，只需要总结原文的内容并回答问题。
    不要提供任何个人观点或者评论。
    不要产生幻觉。
    回答尽可能的简单明了。
    用中文回答问题。

"""
)

merge_results_prompt.input_variables = ["context", "input"]

merge_results_chain = LLMChain(
    llm=Tongyi(model_name="qwen-max", model_kwargs={"temperature": 0.3}),
    prompt=merge_results_prompt,
    verbose=True,
)

def merge_results(inp):
    results, qa_map_path = inp['results'], inp['qa_map_path']
    with open(qa_map_path,'r') as f:
        qa_map = json.load(f)

    print('!!!!!!')
    print(results)
    question_list = [res.strip() for res in results.split('\n') if len(res.strip()) > 0]
    print("length of the answer: ", len(question_list))
    print(question_list)
    merged_answers = '\n\n'.join([qa_map[q] for q in question_list])
    print(merged_answers)
    print("input: ", RunnablePassthrough())
    return {"context": RunnableLambda(lambda x: {"output": merged_answers}), "input": RunnablePassthrough()} | merge_results_chain

def create_atomic_retriever_agent(tools, qa_map_path, system_prompt=None):

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
                print("~~~~~:", model_output)
                return model_output["question"]["input"]
            def _parse_retreiver_outputs(model_output):
                # model_output is a string with top k matches.
                return {'results': model_output, 'qa_map_path': qa_map_path}
            # print("~~~~~:", model_output)
            return _parse_retreiver_inputs | chosen_tool | _parse_retreiver_outputs | RunnableLambda(merge_results) | output_parser_from_merged_results
            # return {"context": _parse_retreiver_inputs | chosen_tool, "input": RunnablePassthrough()} | summarization_llm_prompt | summarization_llm | output_parser
        
        return {"params": RunnableLambda(lambda x: x["tool_use"]["arguments"])} | chosen_tool | output_parser

    runnable = RunnableParallel(
        question=RunnablePassthrough(),
        tool_use=chain
    )
    return runnable | tool_chain


def create_atomic_retriever_agent_single_tool_qa_map(tool, qa_map_path, system_prompt=None):
    """Only run qa map with this function.
    If you don't have extra request from the users but just retrieval, use this one.
    """
    def _parse_retreiver_outputs(model_output):
        # model_output is a string with top k matches.
        return {'results': model_output, 'qa_map_path': qa_map_path}

    return RunnableLambda(lambda x: x["input"]) | tool | _parse_retreiver_outputs | RunnableLambda(merge_results) | output_parser_from_merged_results


def output_parser_from_merged_results(model_output):
    return {"output": model_output['text']}

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