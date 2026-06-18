import os
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma")

SYSTEM_PROMPT = (
    "Ты — аналитический ассистент по исследованию девиантного поведения подростков "
    "в регионах России. Отвечай академическим стилем на русском языке."
)


def get_agent_executor(tools: list, api_key: str | None = None):
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=api_key,
        temperature=0.1,
        max_tokens=2048,
    )

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
