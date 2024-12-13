from typing import List
from fuzzywuzzy import fuzz
import os
import json
import operator
import streamlit as st

from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)
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

    memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")
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


merge_results_prompt = PromptTemplate.from_template(
    """根据提供的信息，回答用户的提问“{input}”。遵循以下步骤：
    1. 参考信息中包含若干个参考问题及其回复，通过匹配参考问题筛选出与用户的问题最相关的问题并获取其回复内容，忽略其他参考内容。
    2. 对选取的回复进行总结并口语化总结，模仿中文人工客服的语气进行回答。

参考内容：{context}

指南：
1. 只需要在参考内容中选择相关的部分进行总结，可以舍弃不相关的部分。
2. 回答模仿人工客服语气，应精简、口语化、有礼貌。
3. 如果提供的内容中没有足够信息解决问题，应明确指出，并提出下一步会对接人工客服。
    """
)

merge_results_prompt.input_variables = ["context", "input"]

merge_results_chain = LLMChain(
    llm=Tongyi(model_name=LLM_NAME, model_kwargs={"temperature": 0.3}),
    prompt=merge_results_prompt,
    verbose=True,
)


def merge_results(inp):
    print("input: ", inp)
    results, qa_map_path, orig_question = (
        inp["results"]["tool_chain"],
        inp["qa_map_path"],
        inp["orig_question"],
    )
    with open(qa_map_path, "r") as f:
        qa_map = json.load(f)

    print(results)
    question_list = [res.strip() for res in results.split("\n") if len(res.strip()) > 0]
    print("length of the answer: ", len(question_list))
    print(question_list)
    merged_answers = "\n\n".join([qa_map[q] for q in question_list])
    print(merged_answers)
    print("input: ", RunnablePassthrough())
    return {
        "context": RunnableLambda(lambda x: {"output": merged_answers}),
        "input": RunnableLambda(lambda x: orig_question),
    } | merge_results_chain


def create_atomic_retriever_agent(tools, qa_map_path, system_prompt=None):

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
                return model_output["question"]["input"]

            def _parse_retreiver_outputs(model_output):
                # model_output is a string with top k matches.
                return {
                    "results": model_output,
                    "qa_map_path": qa_map_path,
                    "orig_question": model_output["question"]["question"]["input"],
                }

            run_chosen_tool = RunnableParallel(
                question=RunnablePassthrough(),
                tool_chain=_parse_retreiver_inputs | chosen_tool,
            )
            return (
                run_chosen_tool
                | _parse_retreiver_outputs
                | RunnableLambda(merge_results)
                | output_parser_from_merged_results
            )

        return (
            {"params": RunnableLambda(lambda x: x["tool_use"]["arguments"])}
            | chosen_tool
            | output_parser
        )

    runnable = RunnableParallel(question=RunnablePassthrough(), tool_use=chain)
    return runnable | tool_chain


def create_atomic_retriever_agent_single_tool_qa_map(
    tool, qa_map_path, system_prompt=None
):
    """Only run qa map with this function.
    If you don't have extra request from the users but just retrieval, use this one.
    """

    def _parse_retreiver_outputs(model_output):
        # model_output is a string with top k matches.
        return {
            "results": model_output,
            "qa_map_path": qa_map_path,
            "orig_question": model_output["question"]["input"],
        }

    run_chosen_tool = RunnableParallel(
        question=RunnablePassthrough(),
        tool_chain=RunnableLambda(lambda x: x["input"]) | tool,
    )

    return (
        run_chosen_tool
        | _parse_retreiver_outputs
        | RunnableLambda(merge_results)
        | output_parser_from_merged_results
    )


def output_parser_from_merged_results(model_output):
    return {"output": model_output["text"]}


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
    return (
        chain
        | {"params": RunnableLambda(lambda x: x["arguments"])}
        | tool
        | output_parser
    )


def check_user_location(user_provided_location: str, locations: List[str]):
    """Check if the user_provided_location overlaps with the items in locations. Return the item with max overlap using fuzzy matching."""
    location_scores = {
        loc: fuzz.ratio(user_provided_location, loc) for loc in locations
    }
    sorted_scores = sorted(
        location_scores.items(), key=operator.itemgetter(1), reverse=True
    )
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
