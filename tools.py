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


