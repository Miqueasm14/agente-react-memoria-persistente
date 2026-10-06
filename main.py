"""Prueba de ejecución del agente ReAct con memoria persistente.

Corre tres conversaciones y guarda la traza completa en traza_ejecucion.json:
  1. Razonamiento multi-paso: el agente llama a herramientas al menos dos veces.
  2. Memoria: una pregunta de seguimiento en el mismo thread_id, que solo se entiende
     con el contexto del turno anterior.
  3. Ciclo de retorno: un cliente inexistente, para ver cómo reacciona ante un ERROR.

Uso: python main.py
"""

import asyncio
import json
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

load_dotenv()

from agent import construir_grafo, get_llm  # noqa: E402  (después de cargar el .env)

DB_PATH = "checkpoints.sqlite"
TRAZA_PATH = "traza_ejecucion.json"
RECURSION_LIMIT = 10  # techo de pasos por invocación: evita bucles infinitos y costos inesperados


def extraer_texto(mensaje: BaseMessage) -> str:
    """Normaliza el contenido de un mensaje (algunos modelos devuelven una lista de bloques)."""
    contenido = mensaje.content
    if isinstance(contenido, str):
        return contenido
    return "".join(b.get("text", "") for b in contenido if isinstance(b, dict))


def serializar(mensajes: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convierte los mensajes de un turno a un formato JSON legible (la traza ReAct)."""
    pasos = []
    for m in mensajes:
        if isinstance(m, HumanMessage):
            pasos.append({"paso": "usuario", "contenido": extraer_texto(m)})
        elif isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                pasos.append({"paso": "el agente decide usar una herramienta", "herramienta": tc["name"], "argumentos": tc["args"]})
        elif isinstance(m, ToolMessage):
            pasos.append({"paso": "la herramienta devuelve", "herramienta": m.name, "resultado": extraer_texto(m)})
        elif isinstance(m, AIMessage):
            pasos.append({"paso": "respuesta final", "contenido": extraer_texto(m)})
    return pasos


def imprimir(pasos: list[dict[str, Any]]) -> None:
    for p in pasos:
        if p["paso"] == "usuario":
            print(f'Usuario: "{p["contenido"]}"')
        elif p["paso"] == "el agente decide usar una herramienta":
            args = ", ".join(f"{k}={v!r}" for k, v in p["argumentos"].items())
            print(f"--> El agente decide usar la herramienta: {p['herramienta']}({args})")
        elif p["paso"] == "la herramienta devuelve":
            print(f"--> La herramienta devuelve: {p['resultado']}")
        else:
            print(f'Respuesta: "{p["contenido"]}"')


async def turno(pregunta: str, thread_id: str) -> list[dict[str, Any]]:
    """Ejecuta un turno de conversación y devuelve solo los pasos nuevos de ese turno.

    Cada turno abre su propia conexión al SQLite: si el agente recuerda la conversación
    es porque la recupera del archivo, no de una variable en memoria.
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app = construir_grafo(get_llm()).compile(checkpointer=checkpointer)
        previos = len((await app.aget_state(config)).values.get("messages", []))
        resultado = await app.ainvoke({"messages": [HumanMessage(pregunta)]}, config=config)

    pasos = serializar(resultado["messages"][previos:])
    print(f"\n[thread_id={thread_id} | mensajes previos recuperados del checkpointer: {previos}]")
    imprimir(pasos)
    return pasos


async def main() -> None:
    # thread_ids únicos por ejecución, para que cada corrida empiece con una sesión limpia.
    sello = datetime.now().strftime("%Y%m%d-%H%M%S")
    hilo_demo = f"demo-{sello}"
    hilo_error = f"error-{sello}"

    print("=" * 80)
    print("1) Razonamiento multi-paso")
    print("=" * 80)
    paso1 = await turno("¿Cuántos pedidos tuvo Ana García y cuál fue el total?", hilo_demo)

    print("\n" + "=" * 80)
    print("2) Memoria: mismo thread_id, pregunta que depende del turno anterior")
    print("=" * 80)
    paso2 = await turno("¿Y Juan Pérez?", hilo_demo)

    print("\n" + "=" * 80)
    print("3) Ciclo de retorno: cliente inexistente")
    print("=" * 80)
    paso3 = await turno("¿Cuántos pedidos tuvo el cliente Roberto Sánchez?", hilo_error)

    traza = {
        "recursion_limit": RECURSION_LIMIT,
        "conversaciones": [
            {"prueba": "razonamiento multi-paso", "thread_id": hilo_demo, "pasos": paso1},
            {"prueba": "memoria (mismo thread_id)", "thread_id": hilo_demo, "pasos": paso2},
            {"prueba": "ciclo de retorno (error de herramienta)", "thread_id": hilo_error, "pasos": paso3},
        ],
    }
    with open(TRAZA_PATH, "w", encoding="utf-8") as f:
        json.dump(traza, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Traza guardada en {TRAZA_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
