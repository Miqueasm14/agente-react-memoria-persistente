"""Herramientas del agente: simulan consultas a una base de datos de clientes y pedidos.

El LLM decide qué herramienta usar leyendo SOLO el nombre y el docstring de cada una,
por eso los docstrings explican con detalle cuándo usarla y qué devuelve.
"""

from langchain_core.tools import tool

# "Base de datos" simulada, en dos tablas separadas: el usuario pregunta por nombre,
# pero los pedidos se buscan por ID. Eso obliga al agente a razonar en dos pasos.
CLIENTES_DB: dict[str, int] = {
    "ana garcía": 102,
    "juan pérez": 205,
    "maría lópez": 310,
}

PEDIDOS_DB: dict[int, dict[str, int]] = {
    102: {"pedidos": 3, "total": 14500},
    205: {"pedidos": 1, "total": 3200},
    310: {"pedidos": 5, "total": 27800},
}


@tool
def buscar_cliente_por_nombre(nombre: str) -> str:
    """Busca el ID interno (cliente_id) de un cliente a partir de su nombre completo.

    Usar esta herramienta SIEMPRE que el usuario mencione a un cliente por su nombre
    y haga falta su cliente_id para consultar sus pedidos.
    Devuelve un mensaje que empieza con ERROR si el nombre no coincide con ningún
    cliente registrado.
    """
    clave = nombre.strip().lower()
    if clave not in CLIENTES_DB:
        return (
            f"ERROR: no se encontró ningún cliente con el nombre '{nombre}'. "
            "Verificá que el nombre esté completo y bien escrito."
        )
    return f"Cliente encontrado: '{nombre}' -> cliente_id={CLIENTES_DB[clave]}"


@tool
def buscar_pedidos(cliente_id: int) -> str:
    """Busca la cantidad de pedidos y el monto total gastado por un cliente,
    dado su cliente_id NUMÉRICO.

    No acepta nombres: si solo se conoce el nombre del cliente, primero usar
    buscar_cliente_por_nombre para obtener su cliente_id.
    Devuelve un mensaje que empieza con ERROR si el cliente_id no existe.
    """
    if cliente_id not in PEDIDOS_DB:
        return f"ERROR: no existe ningún cliente con cliente_id={cliente_id}."
    datos = PEDIDOS_DB[cliente_id]
    return f'{{"pedidos": {datos["pedidos"]}, "total": {datos["total"]}}}'


HERRAMIENTAS = [buscar_cliente_por_nombre, buscar_pedidos]
