"""Agente ReAct: un StateGraph con un nodo de modelo y un nodo de herramientas en ciclo.

El LLM decide en cada turno si llama a una herramienta o responde; el grafo no tiene
rutas if/else manuales: el ruteo lo hace tools_condition según la respuesta del modelo.
"""

import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, trim_messages
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy

from tools import HERRAMIENTAS

# Límite de mensajes del historial que se le envían al LLM en cada llamada. El estado
# completo se sigue guardando en el checkpointer; solo se recorta lo que ve el modelo,
# para que el contexto no crezca sin límite en conversaciones largas ("estado sucio").
MAX_MENSAJES_CONTEXTO = 20

SYSTEM_PROMPT = (
    "Sos un asistente de atención al cliente con acceso a la base de datos de pedidos. "
    "Usá las herramientas para obtener datos reales: nunca inventes IDs, cantidades ni montos. "
    "Si una herramienta devuelve un ERROR o información incompleta, intentá de nuevo "
    "corrigiendo los argumentos, o pedile al usuario que aclare el dato que falta."
)


class EstadoAgente(MessagesState):
    """Estado del agente. Hereda de MessagesState la lista "messages" con el reducer
    add_messages: cada nodo devuelve solo los mensajes nuevos y LangGraph los agrega
    al historial, en lugar de reemplazarlo."""


def get_llm(provider: str | None = None) -> BaseChatModel:
    """Devuelve el chat model del proveedor elegido en LLM_PROVIDER (openai, anthropic o gemini)."""
    provider = (provider or os.environ.get("LLM_PROVIDER", "openai")).lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI  # usa OPENAI_API_KEY
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # usa ANTHROPIC_API_KEY
        return ChatAnthropic(model="claude-opus-5")
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI  # usa GOOGLE_API_KEY
        # Versión "lite": en el plan gratuito tiene más pedidos diarios disponibles que
        # gemini-flash-latest (limitado a 20 por día), y cada corrida de main.py hace ~8.
        return ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)

    raise ValueError(f"LLM_PROVIDER no soportado: '{provider}' (usá openai, anthropic o gemini)")


def construir_grafo(llm: BaseChatModel) -> StateGraph:
    """Arma el grafo ReAct: modelo -> (tools_condition) -> herramientas -> modelo -> ..."""
    llm_con_herramientas = llm.bind_tools(HERRAMIENTAS)

    async def nodo_modelo(state: EstadoAgente) -> dict[str, list]:
        """Envía el historial (recortado) al LLM. La respuesta puede traer tool_calls
        o ser la respuesta final al usuario."""
        historial = trim_messages(
            state["messages"],
            max_tokens=MAX_MENSAJES_CONTEXTO,
            token_counter=len,     # cuenta mensajes, no tokens
            strategy="last",       # conserva los más recientes
            start_on="human",      # no corta un par tool_call / resultado por la mitad
        )
        respuesta = await llm_con_herramientas.ainvoke([SystemMessage(SYSTEM_PROMPT), *historial])
        return {"messages": [respuesta]}

    grafo = StateGraph(EstadoAgente)
    # Si la API del LLM falla de forma transitoria (por ejemplo, un 503 por sobrecarga),
    # LangGraph reintenta el nodo hasta 3 veces en lugar de cortar la ejecución.
    grafo.add_node("modelo", nodo_modelo, retry_policy=RetryPolicy(max_attempts=3, initial_interval=5.0))
    # handle_tool_errors=True: si una herramienta lanza una excepción, el error vuelve
    # al modelo como mensaje (para que reintente o pida aclaración) en vez de cortar el programa.
    grafo.add_node("herramientas", ToolNode(HERRAMIENTAS, handle_tool_errors=True))

    grafo.add_edge(START, "modelo")
    grafo.add_conditional_edges("modelo", tools_condition, {"tools": "herramientas", END: END})
    grafo.add_edge("herramientas", "modelo")  # el ciclo ReAct: el resultado vuelve al modelo
    return grafo
