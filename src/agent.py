import logging
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    room_io,
)
from livekit.plugins import noise_cancellation, silero, elevenlabs, deepgram, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel


logger = logging.getLogger("agent")

load_dotenv(".env.local")

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
        )

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="orsan-v2")
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
