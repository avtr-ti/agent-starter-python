import asyncio
import logging
import os
import random
import tempfile
import threading
import time
from pathlib import Path
from dotenv import load_dotenv
from livekit import rtc, api
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
from livekit.plugins import noise_cancellation, silero, elevenlabs, deepgram, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from openai import OpenAI
from supabase import create_client


logger = logging.getLogger("agent")

load_dotenv(".env.local")

AGENT_NAME = os.getenv("AGENT_NAME")
DB_TABLE_NAME = os.getenv("DB_TABLE_NAME")
DB_SCHEMA = os.getenv("DB_SCHEMA")
DB_RPC_FUNCTION = os.getenv("DB_RPC_FUNCTION")
TARGET_ROOM = os.getenv("ROOM_NAME")

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
                    # Personality
                    You are a helpful and efficient HR support agent.
                    You are polite, patient, and solution-oriented.
                    Eres un asistente de IA útil que proporciona información sobre los procesos de recursos humanos de la empresa. Eres informativo, amable y con enfoque a solución de necesidades de información.
                    # Environment
                    You are assisting employees through a customer support channel.
                    You have access to company information and resources to answer their questions.
                    You may need to access and interpret website data, such as cookie information, to understand the context of their inquiries.
                    The employee may be contacting you with a variety of questions related to HR policies, benefits, or other company-related matters.
                    # Tone
                    Your responses are clear, concise, and professional.
                    You use a friendly and approachable tone.
                    You are patient and understanding, even when dealing with complex or frustrating issues.
                    You avoid using jargon or technical terms that employees may not understand.
                    You speak Spanish.
                    # Goal
                    Your primary goal is to efficiently respond to employee inquiries and direct them to the appropriate person according to their needs if you cannot resolve the issue yourself.
                    1.  **Initial Assessment:**
                        *   Identify the employee's reason for contacting support.
                        *   Determine the urgency and complexity of the issue.
                        *   Gather any necessary information from the employee to understand the issue fully.
                    2.  **Information Retrieval:**
                        *   Search the company's knowledge base or other resources for relevant information.
                        *   Interpret website data, such as cookie information, if necessary, to understand the context of the inquiry.
                        *   If the information is not readily available, consult with other HR staff members.
                    3.  **Resolution or Escalation:**
                        *   If you can resolve the issue yourself, provide the employee with clear and concise instructions or information.
                        *   If you cannot resolve the issue, direct the employee to the appropriate person or department.
                        *   Provide the employee with contact information and any relevant details about the escalation process.
                    4.  **Follow-up:**
                        *   If necessary, follow up with the employee to ensure that their issue has been resolved.
                        *   Document the interaction and any steps taken to resolve the issue.
                    5. Proporcionar información sobre el contenido de las políticas, procedimientos, reglamentos, contratos, procesos y servicios de recursos humanos de la empresa.
                    6. Responder preguntas particulares de cada empelado, sobre los procesos de acuerdo con la información del sistema.
                    7. Dirigir a los colaboradores al web interno del contenido de la información de la empresa https://preprod-oma.intelexion.com/V6OMA/
                    Success is measured by the speed and accuracy of your responses, as well as the employee's satisfaction with the support they receive.
                    # Guardrails
                    Remain within the scope of HR-related inquiries.
                    Do not provide advice on legal or financial matters.
                    Protect employee privacy and confidentiality.
                    Maintain a professional and respectful tone at all times.
                    If you are unsure about an answer, admit that you don't know and offer to find out.
                    Avoid expressing personal opinions or beliefs.
                    •	No proporciones información sobre temas no relacionados con información fuera del sitio.
                    •	Solicitar la clave de confirmación personal del empleado, para acceder a información particular.
                    •	No proporcionar información de otras personas.
                    •	Si no tienes claro una respuesta, indica educadamente que le sugieres contactar a la persona responsable de capital humano para su área y proporcionar sus datos de contacto.
                    •	No hagas promesas que no puedas cumplir. Mantén siempre una actitud profesional y cortés.
                    Cuando transfieras la llamada con un humano notificale a la persona que te conteste en la llamada que un usuario solicito hablar con un humano
                    Si te preguntan por la cantidad de dias de vacaciones que tiene disponible solicita la fecha en que inicio a trabajar el empleado y calcula en base a los datos de vacaciones por antigüedad
                    No requieres ninguna clave de empleado para calcular los dias de vacaciones que tiene disponible
                    # Dias festivos
                    Son los días que la Ley Federal del Trabajo establece como descanso obligatorio: 
                    •	El 1o. de enero
                    •	El primer lunes de febrero en conmemoración del 5 de febrero: día de la Constitución
                    •	El tercer lunes de marzo en conmemoración del 21 de marzo: natalicio de Benito Juárez
                    •	El 1o. de mayo, día del Trabajo
                    •	El 16 de septiembre, día de la Independencia de México
                    •	El tercer lunes de noviembre en conmemoración del 20 de noviembre: día de la Revolución mexicana
                    •	El 1o. de octubre de cada seis años, cuando corresponda a la transmisión del Poder Ejecutivo Federal (atendiendo a la reforma del artículo 83 de la Constitución Política de los Estados Unidos Mexicanos, publicada en el Diario Oficial de la Federación el 10 de febrero de 2014)
                    •	El 25 de diciembre, Navidad
                    •	El que determinen las leyes federales y locales electorales para efectuar la jornada electoral en caso de elecciones ordinarias 
                    Tabla de días de vacaciones por antigüedad:
                    Antigüedad (años)	Días de vacaciones
                    1	12
                    2	14
                    3	16
                    4	18
                    5	20
                    6-10	22
                    11-15	24
                    16-20	26
                    21-25	28
                    26-30	30
                    # Prestaciones
                    Prestaciones de Ley:
                    •	Seguridad Social: Cobertura médica y beneficios a través del IMSS.
                    •	Vacaciones: Conforme a la ley, con derecho a un periodo vacacional anual.
                    •	Prima Vacacional: Un porcentaje adicional al salario base por vacaciones.
                    •	Prima Dominical: Un porcentaje adicional al salario por trabajar en domingo.
                    •	Reparto de Utilidades: Participación de los trabajadores en las ganancias de la empresa. 
                    Prestaciones Adicionales:
                    •	Caja de Ahorro: Un fondo donde los empleados pueden ahorrar y recibir intereses. 
                    •	Bonos Mensuales: Reconocimientos económicos por desempeño. 
                    •	Bonos por Productividad: Incentivos por lograr objetivos de ventas y productividad. 
                    •	Vales de Despensa: Tarjetas electrónicas o vales para comprar alimentos y productos de consumo. 
                    •	Seguro de Vida: Protección económica para los beneficiarios en caso de fallecimiento del empleado. 
                    •	Membresía para Gimnasio: Acceso a instalaciones deportivas. 
                    •	Cursos y Capacitaciones: Oportunidades de desarrollo profesional. 
                    •	Horarios Flexibles: Algunas posiciones ofrecen horarios de trabajo más flexibles. 
                    # Dias de pago
                    •	Los días de pago son los días 14 y 29 de cada mes.
                    •	El fondo de ahorro se paga la primer semana de febrero y la primer semana de agosto.
                    •	El bono anual se paga en dos partes la primer semana de marzo y la ultima semana de septiembre.

                    dia de hoy {{system__time}}
                    # Tools
                    - buscar_informacion_orsan: Busca información relevante sobre OMA Energía en la base de conocimientos.



                """,
            tools=[search_extra_info],
        )

    async def on_enter(self) -> None:
        """Initial message when the agent enters the session."""
        await self.session.generate_reply(
            instructions=(
                "Greet the employee in a friendly and professional manner. "
                "Hola, soy tu asistente virtual de capital humano, con gusto te puedo brindar información sobre los procesos de capital humano de OMA, ¿En qué te puedo ayudar?"
            )
        )

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm

# Lock compartido para evitar múltiples redespachos simultáneos
_redispatch_lock = threading.Lock()

def _get_redispatch_lock_file():
    """Obtiene el archivo de lock para el redespacho."""
    if TARGET_ROOM:
        return Path(tempfile.gettempdir()) / f"livekit_redispatch_{TARGET_ROOM}.lock"
    return None

async def auto_dispatch_to_room():
    """
    Despacha automáticamente el agente a la sala objetivo cuando el servidor se inicia.
    Usa un lock de archivo para evitar múltiples redespachos simultáneos.
    """
    # Intentar adquirir el lock de archivo para evitar múltiples redespachos
    lock_file = _get_redispatch_lock_file()
    lock_acquired = False
    lock_file_handle = None
    
    if not lock_file:
        logger.warning("TARGET_ROOM no está definido, no se puede crear lock de redespacho")
        return
    
    try:
        # Intentar crear/abrir el archivo de lock de forma exclusiva
        try:
            lock_file_handle = lock_file.open('x')
            lock_acquired = True
        except FileExistsError:
            # Si el archivo ya existe, otro proceso ya está redespachando
            logger.debug(f"Otro proceso ya está redespachando a '{TARGET_ROOM}', saltando redespacho")
            return
        
        # Pequeño delay adicional para asegurar que solo un proceso proceda
        await asyncio.sleep(0.2)
        
        lkapi = None
        try:
            lkapi = api.LiveKitAPI(
                url=os.getenv("LIVEKIT_URL"),
                api_key=os.getenv("LIVEKIT_API_KEY"),
                api_secret=os.getenv("LIVEKIT_API_SECRET"),
            )
            
            # Crear dispatch a la sala objetivo
            dispatch = await lkapi.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name=AGENT_NAME,
                    room=TARGET_ROOM,
                )
            )
            logger.info(f"Agente despachado automáticamente a la sala '{TARGET_ROOM}' (Job ID: {dispatch.job_id})")
        except Exception as e:
            # Si el error es que ya existe un dispatch, no es un problema crítico
            error_msg = str(e).lower()
            if 'already' in error_msg or 'exists' in error_msg or 'job_id' in error_msg:
                logger.debug(f"Dispatch ya existe para '{TARGET_ROOM}': {type(e).__name__}")
            else:
                logger.warning(f"No se pudo crear dispatch automático a '{TARGET_ROOM}': {type(e).__name__}: {str(e)}")
                logger.info("El agente esperará a ser despachado manualmente o por reglas de dispatch")
        finally:
            # Asegurar que la sesión HTTP se cierre siempre
            if lkapi is not None:
                try:
                    await lkapi.aclose()
                except Exception as e:
                    logger.debug(f"Error al cerrar sesión de API de LiveKit: {e}")
    finally:
        # Liberar el lock eliminando el archivo
        if lock_acquired and lock_file_handle is not None and lock_file:
            try:
                lock_file_handle.close()
                lock_file.unlink(missing_ok=True)
            except Exception as e:
                logger.debug(f"Error al liberar lock de redespacho: {e}")


@server.rtc_session(agent_name=AGENT_NAME)
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # stt=inference.STT(model="assemblyai/universal-streaming", language="es"),
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

    # Bandera para evitar múltiples redespachos
    redispatch_scheduled = False

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,  # Usar ctx.room que ya está asignado por el dispatch
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
            close_on_disconnect=False,  # Mantener el agente en la room aunque no haya participantes
        ),
    )

    # Handler para detectar cuando se cierra la sesión y redespachar el agente
    def on_session_close(ev):
        """
        Detecta cuando la sesión se cierra y redespacha el agente a la sala.
        """
        nonlocal redispatch_scheduled
        
        # Evitar múltiples redespachos
        if redispatch_scheduled:
            return
        
        redispatch_scheduled = True
        
        logger.info(
            f"Sesión cerrada en la sala '{ctx.room.name}'. "
            f"Razón: {ev.reason}. Redespachando agente a la sala."
        )
        
        # Ejecutar el redespacho en un thread separado para que se complete
        # incluso si el proceso principal está terminando
        def redispatch_in_thread():
            """Ejecuta el redespacho en un thread separado."""
            # Pequeño delay aleatorio para evitar que múltiples procesos redespachen simultáneamente
            time.sleep(random.uniform(0.1, 0.5))
            
            try:
                # Crear un nuevo event loop para este thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Ejecutar el redespacho
                    loop.run_until_complete(auto_dispatch_to_room())
                finally:
                    # Cerrar todas las tareas pendientes y el loop
                    try:
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()
                        if pending:
                            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    except Exception:
                        pass  # Ignorar errores al limpiar
                    finally:
                        loop.close()
            except Exception as e:
                logger.error(f"Error al redespachar agente después de cierre de sesión: {type(e).__name__}: {str(e)}")
        
        # Iniciar el thread como daemon para que no bloquee la salida del proceso
        # El redespacho debería completarse rápidamente (solo una llamada HTTP)
        dispatch_thread = threading.Thread(target=redispatch_in_thread, daemon=True, name="redispatch_thread")
        dispatch_thread.start()
        
        # Dar un momento para que el thread inicie y ejecute el redespacho
        # pero no bloquear indefinidamente
        dispatch_thread.join(timeout=2.0)  # Timeout de 2 segundos

    # Registrar el handler para el evento de cierre de sesión
    session.on("close", on_session_close)

    # Join the room and connect to the user
    await ctx.connect()


async def start_server_with_auto_dispatch():
    """
    Inicia el servidor y crea un dispatch automático a la sala objetivo después de un breve delay.
    """
    # Esperar un momento para que el servidor se registre
    await asyncio.sleep(2)
    
    # Crear dispatch automático a la sala objetivo
    if AGENT_NAME and os.getenv("LIVEKIT_URL") and os.getenv("LIVEKIT_API_KEY"):
        try:
            await auto_dispatch_to_room()
        except Exception as e:
            logger.warning(f"Error al crear dispatch automático: {e}")


if __name__ == "__main__":
    # Iniciar el dispatch automático en segundo plano
    if AGENT_NAME and os.getenv("LIVEKIT_URL") and os.getenv("LIVEKIT_API_KEY"):
        import threading
        
        def run_dispatch():
            """Ejecuta el dispatch automático en un hilo separado."""
            try:
                asyncio.run(start_server_with_auto_dispatch())
            except Exception as e:
                logger.warning(f"Error en hilo de dispatch automático: {e}")
        
        # Iniciar el hilo de dispatch automático
        dispatch_thread = threading.Thread(target=run_dispatch, daemon=True)
        dispatch_thread.start()
    
    # Iniciar el servidor del agente
    cli.run_app(server)
