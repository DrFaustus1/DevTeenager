import json

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage

from .chains.agent_chain import get_agent_executor, EMBED_MODEL, CHROMA_DIR


@csrf_exempt
@require_POST
def ask(request):
    try:
        data = json.loads(request.body)
        question = (data.get("question") or "").strip()
        context  = (data.get("context") or "").strip()
        if not question:
            return JsonResponse({"error": "Вопрос не задан"}, status=400)

        if context:
            question = f"{question}\n\nКонтекст региона:\n{context}"

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="deviance_kb",
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        def search_docs(query: str) -> str:
            docs = retriever.invoke(query)
            return "\n\n".join(d.page_content for d in docs) if docs else "Документы не найдены."

        rag_tool = Tool(
            name="search_documents",
            func=search_docs,
            description=(
                "Поиск по нормативным документам: ФЗ-120, Концепция профилактики "
                "безнадзорности и правонарушений несовершеннолетних."
            ),
        )

        agent = get_agent_executor([rag_tool], api_key=settings.ANTHROPIC_API_KEY)
        result = agent.invoke({"messages": [HumanMessage(content=question)]})
        answer = result["messages"][-1].content
        return JsonResponse({"answer": answer})

    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)
