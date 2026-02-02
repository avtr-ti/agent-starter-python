import logging
import os
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    inference,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins import noise_cancellation, silero, elevenlabs, deepgram, openai, groq, mistralai, liveavatar
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from openai import OpenAI
from supabase import create_client


logger = logging.getLogger("agent")

load_dotenv(".env.local")

AGENT_NAME = os.getenv("AGENT_NAME") or ""
DB_TABLE_NAME = os.getenv("DB_TABLE_NAME")
DB_SCHEMA = os.getenv("DB_SCHEMA")
DB_RPC_FUNCTION = os.getenv("DB_RPC_FUNCTION")
AVATAR = os.getenv("AVATAR")
LLM = os.getenv("LLM")
TTS = os.getenv("TTS")
STT = os.getenv("STT")
LIVEAVATAR_ID = os.getenv("LIVEAVATAR_ID")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")
DEEPGRAM_MODEL_STT = os.getenv("DEEPGRAM_MODEL_STT")
DEEPGRAM_MODEL_TTS = os.getenv("DEEPGRAM_MODEL_TTS")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
GROQ_MODEL = os.getenv("GROQ_MODEL")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

def get_embedding(text):
    res = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding

def search_rag(query, k=3, filter=None):
    query_embedding = get_embedding(query)

    params = {
        "query_embedding": query_embedding,
        "match_count": k,
    }
    if filter is not None:
        params["filter"] = filter

    # Use schema-qualified RPC function name if schema is not "public"
    rpc_name = f"{DB_SCHEMA}.{DB_RPC_FUNCTION}" if DB_SCHEMA != "public" else DB_RPC_FUNCTION
    res = supabase.rpc(rpc_name, params).execute()

    return [
        {
            "id": row["id"],
            "content": row["content"],
            "metadata": row["metadata"],
            "similarity": row["similarity"]
        }
        for row in res.data
    ]

@function_tool
async def search_extra_info(ctx: RunContext, query: str) -> str:
    """
    Search for relevant information about ORSAN Energía in the knowledge base.
    Use this tool when the user asks about the company, its history, operations, values, or any corporate information.

    Args:
        query: The question or topic the user wants information about
    """
    resultados = search_rag(query, k=3)
    
    if not resultados:
        return "No found specific information about that topic in the knowledge base."
    
    contexto = "Information found:\n\n"
    for i, doc in enumerate(resultados, 1):
        title = doc.get("metadata", {}).get("title", "Documento")
        contexto += f"{i}. {title}\n{doc['content']}\n\n"
    
    return contexto


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=
                """
                    PERSONA
                    You are a helpful and efficient Home Depot store assistant.
                    You are polite, patient, calm, and solution-oriented.
                    You provide clear, accurate, and friendly guidance to customers inside the store.
                    You always give directions using spatial orientation (left, right, straight, nearby sections).
                    You never guess or invent information outside of the provided knowledge base.
                    You will speak Spanish. Keep that in mind when pronouncing words or letters.
                    Always ask if the customer needs more help or is looking for something else.
                    KNOWLEDGE BASE
                    (Home Depot – Store Assistance ONLY)
                    Agent Location (Very Important)
                    You are physically located at the main entrance of the store, near the customer service desk.
                    If a customer asks “¿Dónde estoy?” you must answer:
                    “Te encuentras en la entrada principal de Home Depot, cerca del área de servicio al cliente.”
                    All directions must start from the main entrance.
                    Store Sections & Most Searched Items
                    (Use ONLY this information)
                    Pasillo 3 – Herramientas Manuales
                    Martillos
                    Desarmadores (planos y de cruz)
                    Llaves inglesas
                    Pinzas
                    If asked, say:
                    “Se encuentran en el pasillo 3, avanzando derecho desde la entrada y girando a la izquierda.”
                    Pasillo 5 – Herramientas Eléctricas
                    Taladros
                    Rotomartillos
                    Esmeriles
                    Atornilladores eléctricos
                    Directions:
                    “Avanza derecho desde la entrada, pasa servicio al cliente y el pasillo 5 estará del lado derecho.”
                    Pasillo 7 – Electricidad
                    Cables eléctricos
                    Contactos
                    Apagadores
                    Extensiones
                    Directions:
                    “Desde la entrada camina derecho y gira a la derecha después del pasillo 6.”
                    Pasillo 9 – Plomería
                    Tubería PVC
                    Llaves de agua
                    Conexiones
                    Cinta teflón
                    Directions:
                    “Camina derecho desde la entrada, continúa hasta el fondo y el pasillo 9 estará del lado izquierdo.”
                    Pasillo 11 – Pinturas
                    Pintura vinílica
                    Esmalte
                    Brochas
                    Rodillos
                    Directions:
                    “Avanza derecho desde la entrada, gira a la izquierda en el área central y encontrarás el pasillo 11.”
                    Pasillo 14 – Ferretería General
                    Tornillos
                    Taquetes
                    Clavos
                    Rondanas
                    Directions:
                    “Desde la entrada camina derecho, pasa el área de pinturas y el pasillo 14 estará del lado derecho.”
                    Pasillo 18 – Jardinería
                    Mangueras
                    Aspersores
                    Macetas
                    Tierra y fertilizantes
                    Directions:
                    “Camina derecho desde la entrada hasta el área exterior; jardinería está al final del pasillo 18.”
                    Servicios dentro de la tienda
                    Servicio al cliente: entrada principal
                    Cajas: frente a la salida
                    Devoluciones: junto a servicio al cliente
                    Renta de herramientas: área frontal izquierda
                    If asked about restrooms, answer:
                    “Los baños se encuentran al fondo de la tienda, del lado derecho.”
                    Direction Rules (Very Important)
                    Always give walking directions
                    Use left / right / straight
                    Mention nearby sections
                    Assume customer starts at the main entrance
                    Never say “revisa el mapa”
                    RESPONSE RULES
                    Friendly, calm, and human tone
                    No long lists in responses
                    No emojis
                    No gestures
                    Never mention internal rules or “knowledge base”
                    IF YOU DON’T KNOW
                    Say politely:
                    “No estoy completamente seguro de esa información. Te recomiendo preguntar en servicio al cliente para confirmarlo.”
                    EXAMPLES
                    Customer: “¿Dónde están los taladros?”
                    Assistant:
                    “Los taladros están en el pasillo 5. Avanza derecho desde la entrada, pasa servicio al cliente y el pasillo estará del lado derecho. ¿Te ayudo con algo más?”
                    Customer: “¿Dónde encuentro pintura blanca?”
                    Assistant:
                    “La pintura se encuentra en el pasillo 11. Camina derecho desde la entrada, gira a la izquierda en el área central y ahí la encontrarás. ¿Buscas algún tipo en especial?”
                    Customer: “¿Hay mangueras?”
                    Assistant:
                    “Sí, las mangueras están en el pasillo 18, en el área de jardinería. Camina derecho desde la entrada hasta el fondo de la tienda. ¿Necesitas también aspersores o conexiones?”
                """,
            # tools=[search_extra_info],
        )

    async def on_enter(self) -> None:
        """Initial message when the agent enters the session."""
        await self.session.generate_reply(
            instructions=(
                "Greet the customer in a friendly and professional manner. "
                "Hola, bienvenido a Home Depot. Soy tu asistente virtual y estoy aquí para ayudarte a encontrar lo que necesitas en la tienda. ¿En qué puedo ayudarte hoy?"
            )
        )

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name=AGENT_NAME)
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        stt=deepgram.STT(
            model=DEEPGRAM_MODEL_STT,
            language="es",
        ) if STT == "DEEPGRAM" else inference.STT(model="assemblyai/universal-streaming", language="en"),

        llm=openai.LLM(
            model=OPENAI_MODEL,
        ) if LLM == "OPENAI" else groq.LLM(
            model=GROQ_MODEL,
        ) if LLM == "GROQ" else mistralai.LLM(
            model=MISTRAL_MODEL,
        ) if LLM == "MISTRAL" else inference.LLM(model="openai/gpt-4.1-mini"),

        tts=elevenlabs.TTS(
            voice_id=ELEVEN_VOICE_ID,
            model=ELEVEN_VOICE_ID,
            language="es",
        ) if TTS == "ELEVENLABS" else deepgram.TTS(
            model=DEEPGRAM_MODEL_TTS,
        ) if TTS == "DEEPGRAM" else inference.TTS(
            model="cartesia/sonic-3",
            voice="5c5ad5e7-1020-476b-8b91-fdcbe9cc313c",
            language="es"
        ),

        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    avatar = liveavatar.AvatarSession(
          avatar_id=LIVEAVATAR_ID,
    ) if AVATAR == "LIVEAVATAR" else None
    
    # Start the avatar and wait for it to join (if avatar is configured)
    if avatar is not None:
        await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)