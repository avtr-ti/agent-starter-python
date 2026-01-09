from supabase import create_client
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(".env.local")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

def embed_text(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding

docs = [
    (
        "Perfil corporativo de Orsan Energía",
        "Orsan Energía es una empresa dedicada a la distribución y comercialización de combustibles, reconocida por su enfoque en la eficiencia operativa, la innovación tecnológica y el compromiso con el desarrollo sostenible."
    ),
    (
        "Historia y expansión de Orsan Energía",
        "Durante sus primeros años, Orsan operó como un distribuidor regional, pero a partir de 2005 inició un ambicioso plan de expansión que incluyó la apertura de estaciones de servicio propias, la construcción de centros logísticos y la implementación de sistemas avanzados de control y trazabilidad de combustibles. Gracias a esta estrategia, la empresa logró consolidarse como un actor relevante en el mercado energético local."
    ),
    (
        "Operación y red de estaciones",
        "Actualmente, Orsan Energía cuenta con más de 180 estaciones de servicio, una red logística integrada y alianzas estratégicas con proveedores internacionales. Además de gasolina y diésel, la compañía ha diversificado su portafolio incorporando combustibles de menor impacto ambiental y soluciones energéticas para el sector industrial y de transporte."
    ),
    (
        "Innovación y sostenibilidad",
        "Fiel a su visión de futuro, Orsan ha invertido en investigación y desarrollo, impulsando iniciativas orientadas a la eficiencia energética, la reducción de emisiones y la digitalización de la experiencia del cliente. Su cultura corporativa se basa en valores como la transparencia, la seguridad, la innovación y la responsabilidad social."
    ),
    (
        "Visión de futuro",
        "Con más de 25 años de trayectoria, Orsan Energía se proyecta como una empresa preparada para enfrentar los desafíos de la transición energética, manteniendo su compromiso de abastecer de manera confiable y responsable a las comunidades donde opera."
    ),
    (
        "Fundación de Orsan Energía",
        "La compañía fue fundada en 1998 en la ciudad de Puerto Bahía, una zona estratégica del litoral sudamericano, por el ingeniero industrial Ricardo Santelices Orrego, quien identificó la necesidad de modernizar la infraestructura de suministro energético en regiones en crecimiento. Desde sus inicios, Orsan se propuso ofrecer combustibles de alta calidad con estándares superiores de seguridad y servicio."
    ),
]

for title, body in docs:
    text_for_embedding = f"{title}\n\n{body}"
    emb = embed_text(text_for_embedding)

    supabase.table("documents").insert({
        "title": title,
        "body": body,
        "embedding": emb
    }).execute()

