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


st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
# openai.api_key = "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"
st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")
st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Streamlit's open-source Python library!",
        }
    ]

llm = OpenAI(
    # model="gpt-4-turbo-preview",
    model="gpt-3.5-turbo-0613",
    temperature=0.2,
    system_prompt="你是一个关于大众云学的专家，你了解关于大众云学的所有问题。假设所有的问题都与大众云学有关。保持你的答案技术性和基于事实——不要产生幻觉。",
    # system_prompt="you are an expert on the Dazhong Cloud Learning platform and your job is to answer technical questions. Assume that all questions are related to the Dazhong Cloud Learning platform. Keep your answers technical and based on facts – do not hallucinate features.",
)


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(
        text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."
    ):
        reader = SimpleDirectoryReader(input_dir="./policies", recursive=True)
        docs = reader.load_data()
        # service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features."))
        service_context = ServiceContext.from_defaults(llm=llm)
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index


index = load_data()


def lookup_by_id():
    return "looking up"


def multiply(a: int, b: int) -> int:
    """计算两个数的乘积并返回结果"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)

policy_engine = index.as_query_engine(similarity_top_k=5, verbose=True)

tools = [
    QueryEngineTool(
        query_engine=policy_engine,
        metadata=ToolMetadata(
            name="policy_engine",
            # description="查询大众云学使用条款及方法，具体关于如何注册和如何查询证书和学时等问题，返回最相关的文档。",
            description="you are an expert on the Dazhong Cloud Learning platform and your job is to answer technical questions. Assume that all questions are related to the Dazhong Cloud Learning platform. Keep your answers technical and based on facts – do not hallucinate features.",
        ),
    ),
    multiply_tool,
]

agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)

# response = agent.chat("你好，我想知道大众云学的使用条款及方法。")
# response = agent.chat("155乘以203等于多少")
# print(str(response))
# exit()
if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    # st.session_state.chat_engine = index.as_chat_engine(
    #     chat_mode="condense_question", verbose=True
    # )
    st.session_state.chat_engine = agent

if prompt := st.chat_input(
    "Your question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history
