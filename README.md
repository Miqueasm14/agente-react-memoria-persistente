# Agente ReAct con memoria persistente (LangGraph)

Agente que razona en ciclo ReAct (*Reasoning + Acting*): en cada paso, el propio LLM decide si necesita llamar a una herramienta o si ya puede responder. Las herramientas simulan consultas a una base de datos de clientes y pedidos, y la conversación se guarda en SQLite: con el mismo `thread_id`, el agente recuerda lo que se habló antes.

## Ejemplo de traza ReAct

Salida real de `python main.py` (la traza completa está en [`traza_ejecucion.json`](traza_ejecucion.json)):

```
[thread_id=demo-20261005-215042 | mensajes previos recuperados del checkpointer: 0]
Usuario: "¿Cuántos pedidos tuvo Ana García y cuál fue el total?"
--> El agente decide usar la herramienta: buscar_cliente_por_nombre(nombre='Ana García')
--> La herramienta devuelve: Cliente encontrado: 'Ana García' -> cliente_id=102
--> El agente decide usar la herramienta: buscar_pedidos(cliente_id=102)
--> La herramienta devuelve: {"pedidos": 3, "total": 14500}
Respuesta: "Ana García tuvo un total de 3 pedidos y el monto total gastado fue de $14.500."

[thread_id=demo-20261005-215042 | mensajes previos recuperados del checkpointer: 6]
Usuario: "¿Y Juan Pérez?"
--> El agente decide usar la herramienta: buscar_cliente_por_nombre(nombre='Juan Pérez')
--> La herramienta devuelve: Cliente encontrado: 'Juan Pérez' -> cliente_id=205
--> El agente decide usar la herramienta: buscar_pedidos(cliente_id=205)
--> La herramienta devuelve: {"pedidos": 1, "total": 3200}
Respuesta: "Juan Pérez tuvo 1 pedido y el monto total gastado fue de $3.200."

[thread_id=error-20261005-215042 | mensajes previos recuperados del checkpointer: 0]
Usuario: "¿Cuántos pedidos tuvo el cliente Roberto Sánchez?"
--> El agente decide usar la herramienta: buscar_cliente_por_nombre(nombre='Roberto Sánchez')
--> La herramienta devuelve: ERROR: no se encontró ningún cliente con el nombre 'Roberto Sánchez'. Verificá que el nombre esté completo y bien escrito.
Respuesta: "No pude encontrar ningún cliente registrado con el nombre "Roberto Sánchez". ¿Podrías verificar si está bien escrito o si tiene algún otro nombre con el que esté registrado?"
```

Las tres pruebas muestran:

1. **Razonamiento multi-paso:** el usuario da un nombre, pero los pedidos se buscan por ID. El agente decide solo llamar primero a `buscar_cliente_por_nombre` y después a `buscar_pedidos`, con el ID que obtuvo: dos llamadas a herramientas para una sola pregunta.
2. **Memoria:** "¿Y Juan Pérez?" no tiene sentido por sí sola. Con el mismo `thread_id`, el agente recupera los 6 mensajes del turno anterior desde SQLite y entiende que se pregunta por pedidos y total.
3. **Ciclo de retorno:** la herramienta devuelve un `ERROR` porque el cliente no existe. El agente no inventa datos: le pide al usuario que verifique el nombre.

## Estructura del repositorio

```
agente-react-memoria-persistente/
├── tools.py               # "Base de datos" simulada y herramientas con @tool
├── agent.py               # Estado (hereda de MessagesState), LLM con bind_tools() y StateGraph con el ciclo ReAct
├── main.py                # Prueba de ejecución con AsyncSqliteSaver, thread_id y recursion_limit
├── traza_ejecucion.json   # Traza ReAct generada por main.py
├── requirements.txt
├── .env.example
└── .gitignore
```

## Cómo funciona

| Pedido de la consigna | Dónde está |
|---|---|
| `StateGraph` que hereda de `MessagesState` | `agent.py` → `EstadoAgente(MessagesState)` y `construir_grafo()` |
| Nodo de modelo + nodo de herramientas + arista condicional (`tools_condition`) | `agent.py` |
| Herramienta propia con `@tool` y docstring descriptivo | `tools.py` |
| LLM vinculado a las herramientas (`llm.bind_tools()`) | `agent.py` → `construir_grafo()` |
| Persistencia con `SqliteSaver` + `thread_id` | `main.py` → `turno()` |
| Prueba multi-paso (herramientas invocadas 2 o más veces) y `recursion_limit` | `main.py` |
| Traza de ejecución en `.json` | `traza_ejecucion.json` |

### 1. Herramientas (`tools.py`)

Dos herramientas con el decorador `@tool`, que simulan consultas a una base de datos:

- `buscar_cliente_por_nombre(nombre)` devuelve el `cliente_id` de un cliente.
- `buscar_pedidos(cliente_id)` devuelve la cantidad de pedidos y el total gastado.

Los datos están en dos "tablas" separadas a propósito: como el usuario pregunta por nombre y los pedidos se buscan por ID, el agente tiene que encadenar las dos herramientas.

El LLM elige qué herramienta usar leyendo **solo su nombre y su docstring**. Por eso cada docstring explica cuándo usarla, qué recibe y qué devuelve, por ejemplo: "si solo se conoce el nombre del cliente, primero usar buscar_cliente_por_nombre". Un docstring vago es la causa más común de que el agente no use la herramienta esperada. Cuando no encuentran el dato, las herramientas devuelven un texto que empieza con `ERROR`, en lugar de lanzar una excepción, para que el modelo lo lea y reaccione.

### 2. LLM y grafo (`agent.py`)

- **LLM:** `get_llm()` crea el modelo según la variable `LLM_PROVIDER` y `bind_tools()` le da acceso a las herramientas.

  | `LLM_PROVIDER` | Modelo | Key necesaria |
  |---|---|---|
  | `openai` (por defecto) | `gpt-4o-mini` | `OPENAI_API_KEY` |
  | `anthropic` | `claude-opus-5` | `ANTHROPIC_API_KEY` |
  | `gemini` | `gemini-flash-lite-latest` | `GOOGLE_API_KEY` |

- **Estado:** `EstadoAgente` hereda de `MessagesState`, que define la lista `messages` con el reducer `add_messages`. El estado no se modifica directamente: cada nodo devuelve solo los mensajes nuevos, y el reducer los agrega al historial. El grafo se crea con `StateGraph(EstadoAgente)`.
- **Grafo:**

  ```
  START → modelo ──(tools_condition)──→ herramientas
            ↑   └──(sin tool_calls)──→ END    │
            └─────────────────────────────────┘
  ```

  `tools_condition` revisa si la respuesta del modelo trae llamadas a herramientas: si las trae, va al nodo `herramientas` (`ToolNode`); si no, termina. La arista `herramientas → modelo` cierra el ciclo: el modelo recibe el resultado y vuelve a decidir. No hay rutas `if/else` manuales: el que decide es el LLM.
- **Ciclo de retorno:** el prompt de sistema le indica al modelo que, si una herramienta devuelve un `ERROR` o datos incompletos, reintente corrigiendo los argumentos o pida una aclaración. `ToolNode(..., handle_tool_errors=True)` hace que, si una herramienta lanza una excepción, el error vuelva al modelo como mensaje en lugar de cortar el programa.
- **Estado sucio:** el historial completo se guarda en el checkpointer, pero al LLM se le envían como máximo los últimos 20 mensajes (`trim_messages`), así el contexto no crece sin límite en conversaciones largas.
- **Errores transitorios de la API:** el nodo del modelo tiene una `RetryPolicy` de LangGraph que lo reintenta hasta 3 veces ante fallas transitorias, como un 503 por sobrecarga.

### 3. Persistencia y prueba de ejecución (`main.py`)

- **Checkpointer:** se usa `AsyncSqliteSaver`, la versión asíncrona de `SqliteSaver` del paquete `langgraph-checkpoint-sqlite`. Es la que corresponde porque el agente se ejecuta con `asyncio` (`ainvoke`). El historial de cada conversación se guarda en `checkpoints.sqlite`, identificado por su `thread_id`.
- **Memoria real:** cada turno abre su propia conexión a SQLite. Si el agente recuerda el turno anterior es porque lo leyó del archivo, no de una variable en memoria; la traza muestra cuántos mensajes previos se recuperaron.
- **`recursion_limit=10`:** techo de pasos del grafo por invocación. Evita bucles infinitos entre el modelo y las herramientas, y costos inesperados en la API.
- **`thread_id` únicos por ejecución:** cada corrida de `main.py` usa IDs nuevos (con fecha y hora), así siempre empieza con sesiones limpias.
- Al terminar, guarda la traza de las tres pruebas en `traza_ejecucion.json`.

## Cómo levantar el entorno y ejecutarlo

Requiere Python 3.12 o superior.

**PowerShell (Windows):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si PowerShell bloquea la activación del entorno virtual, corré una sola vez:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

**Git Bash / Linux / macOS:**
```bash
python -m venv .venv
source .venv/Scripts/activate    # en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**Configurar y ejecutar:**
```bash
cp .env.example .env
# elegir LLM_PROVIDER y completar la API key de ese proveedor

python main.py
```

`main.py` corre las tres pruebas, muestra la traza en consola y reescribe `traza_ejecucion.json`.

## Variables de entorno

| Variable | Para qué se usa |
|---|---|
| `LLM_PROVIDER` | Proveedor del LLM: `openai`, `anthropic` o `gemini` (por defecto `openai`) |
| `OPENAI_API_KEY` | Solo si `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | Solo si `LLM_PROVIDER=anthropic` |
| `GOOGLE_API_KEY` | Solo si `LLM_PROVIDER=gemini` (gratis en [aistudio.google.com/apikey](https://aistudio.google.com/apikey)) |

El `.env` y la base `checkpoints.sqlite` están en `.gitignore`: no se suben al repositorio.

## Notas

- La traza incluida se generó con `LLM_PROVIDER=gemini` (`gemini-flash-lite-latest`). Con el plan gratuito, cada modelo de Gemini tiene un límite de pedidos por día; cada corrida de `main.py` hace alrededor de 8. Si aparece un error `429 RESOURCE_EXHAUSTED`, se agotó la cuota diaria de ese modelo.
- Al usar Gemini, la librería muestra un aviso sobre "automatic function calling (AFC)". Es informativo y no afecta al agente.
