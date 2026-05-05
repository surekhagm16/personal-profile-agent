"""
rag_agent.py — LangGraph RAG Agent
Uses HuggingFace Inference API (Qwen2.5-7B-Instruct) + FAISS for retrieval.
"""

import os
from typing import TypedDict, List, Annotated
import operator

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import (
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
    ChatHuggingFace,
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# ── State ────────────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    question: str
    context: Annotated[List[Document], operator.add]
    answer: str
    chat_history: List[dict]


# ── Build vector store ────────────────────────────────────────────────────────

profile_path = os.path.join(os.path.dirname(__file__), "myprofile.txt")


def build_vectorstore(profile_path: str) -> FAISS:
    """Load profile doc, split, embed, and return FAISS index."""
    loader = TextLoader(profile_path, encoding="utf-8")
    docs = loader.load()
    len(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    len(chunks)
    embeddings = HuggingFaceEmbeddings(model_kwargs={"token": hf_token})
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ── LLM ──────────────────────────────────────────────────────────────────────


llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct", task="text-generation")
model = ChatHuggingFace(llm=llm)


# ── Graph nodes ───────────────────────────────────────────────────────────────

SYSTEM_PERSONA = """You are a friendly and professional personal assistant representing {name}.
Answer questions about {name} based ONLY on the context provided below. Don't make up anything.
If the context does not contain enough information to answer, say so politely.
Keep answers concise, warm, and in first person when appropriate (as if {name} is speaking).
"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_PERSONA
            + """

Context:
{context}

Chat History:
{chat_history}""",
        ),
        ("human", "{question}"),
    ]
)

PERSONA_NAME = "Surekha Madival"  # ← Replace with your name when you update the profile


def generate_answer(state: AgentState) -> AgentState:
    """Generate answer using retrieved context."""
    context_text = "\n\n".join(doc.page_content for doc in state["context"])

    # Format chat history for the prompt
    history_text = ""
    for msg in state.get("chat_history", [])[-6:]:  # last 3 turns
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    formatted_prompt = RAG_PROMPT.format_messages(
        name=PERSONA_NAME,
        context=context_text,
        chat_history=history_text or "None yet.",
        question=state["question"],
    )

    response = model.invoke(formatted_prompt)
    answer = response.content

    # Strip any prompt echo that some models return
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    return {"answer": answer}


# ── Assemble graph ────────────────────────────────────────────────────────────


def build_graph(profile_path: str):
    """Build and compile the LangGraph RAG agent."""
    vectorstore = build_vectorstore(profile_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def retrieve(state: AgentState) -> AgentState:
        """Retrieve relevant chunks from the vector store."""
        docs = retriever.invoke(state["question"])
        return {"context": docs}

    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate_answer)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
