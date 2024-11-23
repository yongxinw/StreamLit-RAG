from typing import List
from fuzzywuzzy import fuzz
import os
import json
import operator
import streamlit as st

from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.dashscope import DashScopeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent

from langchain.tools import BaseTool, StructuredTool, tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.llms import Tongyi
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from statics import DASHSCOPE_API_KEY, LLM_NAME


os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

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
        Tongyi(model_name=LLM_NAME, model_kwargs={"temperature": 0.3}),
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
        Tongyi(model_name=LLM_NAME, model_kwargs={"temperature": 0.3}),
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
# 5. 如果用户的问题与提供的信息无关，请直接回答“不好意思，暂时没有找到相关问题的答案，正在为您转接人工客服。。。”。

merge_results_prompt = PromptTemplate.from_template(
#     """Based on the information provided, directly answer the user's question "{input}". Follow these steps:
# 1. The reference content includes several reference questions and their responses. Identify the most relevant question related to the user's query and retrieve its response, ignoring all other content.
# 2. Summarize this one response and make the summary conversational, ignoring other reference content.

# Reference Content: {context}

# Guidelines:
# 1. Carefully read the user's question to understand the specific requirements.
# 2. Identify only the most relevant response.
# 3. The answer should mimic a human customer agent tone, be concise, conversational, polite, and focused on addressing the issue.
# 4. Use clear Chinese.
# 5. If the provided content does not contain sufficient information to resolve the question, clearly state this and suggest that the next step will be to connect with human customer service.
# 6. Don't hallucinate.

# Example:
# Question: “专技个人登录时，提示验证失败”
# Reference Content: “问题：登录为什么提示验证失败    回复：请不要使用电脑自带和IE浏览器，建议使用谷歌、360浏览器（极速模式）、搜狗等浏览器，正常页面是可以拖动向右箭头完成滑块验证的。

# 问题：审核中，提示‘您的账号正在审核中，审核通过才可登录平台    回复：您可根据提示点击‘查看审核部门’，输入姓名、身份证号、密码进行查看待审核信息，需要审核通过后才可以登录。”

# Thinking steps:
# 1. The most relavant reference question is "登录为什么提示验证失败"
# 2. The most relavant reference question's response is "请不要使用电脑自带和IE浏览器，建议使用谷歌、360浏览器（极速模式）、搜狗等浏览器，正常页面是可以拖动向右箭头完成滑块验证的。"
# 3. The output answer should be a verbal summarization of "请不要使用电脑自带和IE浏览器，建议使用谷歌、360浏览器（极速模式）、搜狗等浏览器，正常页面是可以拖动向右箭头完成滑块验证的。"
# 4. The answer should be “您好，这个问题通常是因为使用电脑自带和IE浏览器，建议你可以使用谷歌、360浏览器（极速模式）、搜狗等浏览器，正常页面是可以拖动向右箭头完成滑块验证的。” 
#     """
# 2. 从参考内容中选择并总结只与问题直接相关的信息，参考内容中一定存在不相关的部分！

    """根据提供的信息，回答用户的提问“{input}”。遵循以下步骤：
    1. 参考信息中包含若干个参考问题及其回复，通过匹配参考问题筛选出与用户的问题最相关的问题并获取其回复内容，忽略其他参考内容。
    2. 对选取的回复进行总结并口语化总结，模仿中文人工客服的语气进行回答。

参考内容：{context}

指南：
1. 只需要在参考内容中选择相关的部分进行总结，可以舍弃不相关的部分。
2. 回答模仿人工客服语气，应精简、口语化、有礼貌。
3. 如果提供的内容中没有足够信息解决问题，应明确指出，并提出下一步会对接人工客服。
    """

# 例如：
# 问题：“专技个人登录时，提示验证失败”
# 参考内容：“问题：登录为什么提示验证失败    回复：请不要使用电脑自带和IE浏览器，建议使用谷歌、360浏览器（极速模式）、搜狗等浏览器，正常页面是可以拖动向右箭头完成滑块验证的。

# 问题：审核中，提示‘您的账号正在审核中，审核通过才可登录平台  回复：您可根据提示点击‘查看审核部门’，输入姓名、身份证号、密码进行查看待审核信息，需要审核通过后才可以登录。”

# 回答应该是：“请不要使用电脑自带和IE浏览器，建议使用谷歌、360浏览器（极速模式）、搜狗等浏览器，正常页面是可以拖动向右箭头完成滑块验证的。”
    
#     """ 你的任务是根据以下内容，用中文总结并回答用户的问题 “{input}”。
#     如果用户提供了反馈或建议，请从参考内容中提取最相关的回复话术，总结并回复。

#     参考内容：{context}
    
#     只需要在参考内容中选择相关的部分进行总结，可以舍弃不相关的部分。
#     不要添加任何新的信息，只需要总结原文的内容并回答问题。
#     不要提供任何个人观点或者评论。
#     不要产生幻觉。
#     回答尽可能的简单明了。
#     用中文回答问题。
# """
)

merge_results_prompt.input_variables = ["context", "input"]

merge_results_chain = LLMChain(
    llm=Tongyi(model_name=LLM_NAME, model_kwargs={"temperature": 0.3}),
    prompt=merge_results_prompt,
    verbose=True,
)

def merge_results(inp):
    print('!!!!!!!!!!!!!!!input: ', inp)
    results, qa_map_path, orig_question = inp['results']['tool_chain'], inp['qa_map_path'], inp['orig_question']
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
    return {"context": RunnableLambda(lambda x: {"output": merged_answers}), "input": RunnableLambda(lambda x: orig_question)} | merge_results_chain

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
    
    model = Tongyi(model_name=LLM_NAME, model_kwargs={"temperature": 0.3})
    chain = prompt | model | JsonOutputParser()

    def tool_chain(model_output):
        tool_map = {tool.name: tool for tool in tools}
        print(model_output)
        chosen_tool = tool_map[model_output["tool_use"]["name"]]
        if "qa_engine" in chosen_tool.name:
            def _parse_retreiver_inputs(model_output):
                print("~~~~~1:", model_output)
                return model_output["question"]["input"]
            def _parse_retreiver_outputs(model_output):
                # model_output is a string with top k matches.
                print("~~~~~2:", print(json.dumps(model_output, indent=4)))
                return {'results': model_output, 'qa_map_path': qa_map_path, 'orig_question': model_output['question']['question']['input']}
            # print("~~~~~:", model_output)
            run_chosen_tool = RunnableParallel(
                question=RunnablePassthrough(),
                tool_chain=_parse_retreiver_inputs | chosen_tool)
            return run_chosen_tool | _parse_retreiver_outputs | RunnableLambda(merge_results) | output_parser_from_merged_results
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
        print("~~~~~22:", model_output)
        return {'results': model_output, 'qa_map_path': qa_map_path, 'orig_question': model_output['question']['input']}
    run_chosen_tool = RunnableParallel(
        question=RunnablePassthrough(),
        tool_chain=RunnableLambda(lambda x: x["input"]) | tool)

    return run_chosen_tool | _parse_retreiver_outputs | RunnableLambda(merge_results) | output_parser_from_merged_results


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

    model = Tongyi(model_name=LLM_NAME, model_kwargs={"temperature": 0.3})
    chain = prompt | model | JsonOutputParser()
    return chain | {"params": RunnableLambda(lambda x: x["arguments"])} | tool | output_parser

def check_user_location(user_provided_location: str, locations: List[str]):
    """ Check if the user_provided_location overlaps with the items in locations. Return the item with max overlap using fuzzy matching.
    """
    location_scores = {loc: fuzz.ratio(user_provided_location, loc) for loc in locations}
    sorted_scores = sorted(location_scores.items(), key=operator.itemgetter(1), reverse=True)
    filtered_scores = [(loc, score) for loc, score in sorted_scores if score > 0]
    if len(filtered_scores) == 0:
        return None
    return filtered_scores[0][0]


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