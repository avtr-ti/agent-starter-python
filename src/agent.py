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
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins import noise_cancellation, silero, elevenlabs, deepgram, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from openai import OpenAI
from supabase import create_client


logger = logging.getLogger("agent")

load_dotenv(".env.local")

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

def search_rag(query, k=3, match_threshold=0.5):
    query_embedding = get_embedding(query)

    res = supabase.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_count": k,
        "match_threshold": match_threshold
    }).execute()

    return [
        {
            "title": row["title"],
            "body": row["body"],
            "similarity": row["similarity"]
        }
        for row in res.data
    ]

@function_tool
async def buscar_informacion_orsan(ctx: RunContext, consulta: str) -> str:
    """
    Busca información relevante sobre ORSAN Energía en la base de conocimientos.
    Usa esta herramienta cuando el usuario pregunte sobre la empresa, su historia, operaciones, valores o cualquier información corporativa.
    
    Args:
        consulta: La pregunta o tema sobre el que el usuario quiere información
    """
    resultados = search_rag(consulta, k=3)
    
    if not resultados:
        return "No encontré información específica sobre ese tema en la base de conocimientos."
    
    contexto = "Información encontrada:\n\n"
    for i, doc in enumerate(resultados, 1):
        contexto += f"{i}. {doc['title']}\n{doc['body']}\n\n"
    
    return contexto


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=
                """
                    Eres el Asistente Virtual de ORSAN. Guías a clientes en el proceso de cargar gasolina paso a paso.

                  REGLAS CRÍTICAS:
                  - Máximo 2 oraciones por respuesta
                  - Sin emojis
                  - Un paso a la vez, espera confirmación del usuario antes de continuar
                  - Nunca des el proceso completo de golpe
                  - Solo avanza cuando el usuario lo pida explícitamente
                  - Cuando el usuario pregunte sobre la empresa ORSAN, su historia, operaciones o información corporativa, usa la herramienta buscar_informacion_orsan para obtener información precisa

                  TONO:
                  - Cordial y profesional
                  - Lenguaje simple y directo
                  - Español neutro

                  PASOS DE CARGA (dar UNO a la vez):

                  1. Estaciona junto a la estación de recarga de gasolina, tanque del lado de la manguera. Apaga el motor.

                  2. Abre la tapa del tanque (busca la palanca interna o ábrela manualmente).

                  3. Elige pago: inserta tarjeta o ve a caja si pagas efectivo. Selecciona monto y tipo de gasolina.

                  4. Retira la manguera de la estación de recarga de gasolina del dispensador e inserta la boquilla en el tanque.

                  5. Aprieta la manguera de gasolina. La estación de recarga de gasolina se detendrá sola cuando termine.

                  6. Retira la manguera de la estación de recarga de gasolina, devuélvela a su lugar y cierra la tapa del tanque.

                  7. Recoge tu recibo si lo necesitas. Listo para partir.

                  EJEMPLO:
                  Usuario: "Es mi primera vez"
                  Asistente: "Perfecto. Paso 1: Estaciona tu auto junto a la estación de recarga de gasolina con el tanque del lado de la manguera y apaga el motor. ¿Listo?"

                  Usuario: "Listo"
                  Asistente: "Paso 2: Abre la tapa del tanque. ¿Ya la abriste?"

                """,
            tools=[buscar_informacion_orsan],
        )

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="orsan-v3")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        stt=deepgram.STT(
            model="nova-3",
            language="es",
        ),
        # llm=inference.LLM(model="openai/gpt-4.1-mini"),
        llm=openai.LLM(
            model="gpt-4o-mini"
        ),
        # tts=inference.TTS(
        #     model="cartesia/sonic-3",
        #     voice="5c5ad5e7-1020-476b-8b91-fdcbe9cc313c",
        #     language="es"
        # ),
        tts=elevenlabs.TTS(
            voice_id="EXAVITQu4vr4xnSDxMaL",
            model="eleven_multilingual_v2",
            language="es",
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # avatar = liveavatar.AvatarSession(
    #   avatar_id="bf00036b-558a-44b5-b2ff-1e3cec0f4ceb",  # ID of the LiveAvatar avatar to use
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)


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
