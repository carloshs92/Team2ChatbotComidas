from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import unicodedata
import firebase_admin
from firebase_admin import credentials, firestore
from twilio.rest import Client
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import os

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")


def load_model(model_name: str):
    try:
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        model = PeftModel.from_pretrained(base_model, model_name)
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None, None

def testConversation():
    # Cargar el modelo y el tokenizador
    model, tokenizer = load_model("carloshuamani/Llama-3.2-ComidaPeruana")
    
    # Validacion de respuestas del modelo
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Se cordial, responde de manera formal al comensal. Reecuerda eres un vendedor de comida peruana y solo de comida peruana, no vendes otra cosa más.

    ### Prompt:
    {}

    ### Response:
    {}"""

    questions = [
    "¿Qué tipo de platos ofreces?",
    "¿Qué platos típicos me recomiendas?",
    "¿Cuáles son los más populares?",
    "¿Qué platos son los más picantes?",
    "¿Vende sushi?",
    "¿Tiene tacos a la paella?",
    "¿Con que bebida me recomiendas acompañar mi ceviche?",
    "¿Que alimentos contiene papa?",
    "¿Que postres tipicos tiene?",
    "¿Vende relojes?",
    "¿Tiene menu para niños de 3 años?",
    "¿Vende camisas?",
    "¿Que platos internacionales tiene?"
    ]

    prompts = [alpaca_prompt.format(question, "") for question in questions]

    # Tokenización correcta con truncamiento para seguridad
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

    # Generar respuestas más controladas
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,         # Respuestas más cortas
        do_sample=False,           # Desactiva muestreo para respuestas más deterministas
        temperature=0.6,           # Menor aleatoriedad
        top_p=0.8,                 # Reducir diversidad
        repetition_penalty=1.2,    # Evitar repeticiones
    )

    # Decodificar las respuestas generadas
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Imprimir las respuestas formateadas
    for i, full_text in enumerate(decoded_outputs):
        # Extraer sólo la parte de la respuesta
        if "### Response:" in full_text:
            response = full_text.split("### Response:")[-1].strip()
        else:
            response = full_text.strip()

        print(f"Pregunta: {questions[i]}")
        print(f"Respuesta: {response}\n{'-'*50}")
        
        return model, tokenizer
    
def connectDatabase():
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate("/db.json")
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None

def initTwilioClient():
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        return client
    except Exception as e:
        print(f"Error al inicializar el cliente de Twilio: {e}")
        return None

def utils(db):
    def obtener_contexto():
        return (
            "Eres un asistente virtual de un restaurante de comida peruana, un restaurante especializado en platos típicos peruanos. "
            "Estás capacitado para responder preguntas sobre nuestros platos, productos y promociones. "
            "Responde siempre de manera amable, clara y profesional."
        )

    def normalizar(texto):
        texto = texto.lower()
        texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
        return texto

    def extraer_plato(mensaje):
        mensaje_norm = normalizar(mensaje)
        platos_ref = db.collection("Platos").stream()

        for doc in platos_ref:
            nombre_plato = doc.to_dict().get("Nombre plato")
            nombre_norm = normalizar(nombre_plato)

            if nombre_norm in mensaje_norm:
                return doc.id
        return None

    def buscar_precio(plato):
        ref = db.collection("Platos").document(plato)
        doc = ref.get()

        if doc.exists:
            data = doc.to_dict()
            producto = data.get("Nombre plato")
            precio = data.get("Precio")

            return f"Claro, el {producto} cuesta S/{precio}."
        else:
            return None

    def extraer_promocion(mensaje):
        mensaje_norm = normalizar(mensaje)
        promocion_ref = db.collection("Promociones").stream()

        for doc in promocion_ref:
            nombre_promocion = doc.to_dict().get("Titulo")
            nombre_norm = normalizar(nombre_promocion)

            if nombre_norm in mensaje_norm:
                return doc.id
        return None

    def buscar_promociones(promocion):
        ref = db.collection("Promociones").document(promocion)
        doc = ref.get()

        if doc.exists:
            data = doc.to_dict()
            titulo = data.get("Titulo")
            descripcion = data.get("Descripcion")

            return f"Claro, tenemos la promocion llamada {titulo}, la cual consiste en {descripcion} ."
        else:
            return None
    return obtener_contexto, normalizar, extraer_plato, buscar_precio, extraer_promocion, buscar_promociones
    
def init():
    model, tokenizer = load_model("carloshuamani/Llama-3.2-ComidaPeruana")
    db = connectDatabase()
    utils_functions = utils(db)
    twilio_client = initTwilioClient()  # Inicializar el cliente de Twilio aquí
    return model, tokenizer, db, utils_functions, twilio_client # Devolver el cliente


app = Flask(__name__)

@app.route('/whatsapp', methods=['POST'])
def whatsapp_mymessage():
    model, tokenizer,  utils_functions = init() # Obtener el cliente
    incoming_msg = request.values.get('Body', '').lower()
    print("Mensaje recibido:", incoming_msg)

    respuesta = ""
    contexto = utils_functions.obtener_contexto()

    promociones_keywords = ["que promociones", "descuentos", "ofertas"]
    precio_keywords = ["precio", "cuánto cuesta", "cual es el costo", "cuanto vale"]

    try:
        if any(keyword in incoming_msg for keyword in precio_keywords):
            producto = utils_functions.extraer_plato(incoming_msg)
            if producto:
                respuesta = utils_functions.buscar_precio(producto)
                if not respuesta:
                    respuesta = f"Lo siento, no encontré detalles para el plato '{producto}'."
            else:
                respuesta = "No contamos con ese plato ¿Puedes pedir otro?"
        elif any(keyword in incoming_msg for keyword in promociones_keywords):
            promocion = utils_functions.extraer_promocion(incoming_msg)
            if promocion:
                respuesta = utils_functions.buscar_promociones(promocion)
                if not respuesta:
                    respuesta = f"Lo siento, no encontré detalles de la promocion."
            else:
                respuesta = "No hay promocion para ello"
        else:
            prompt = (
                f"{contexto}\n\n"
                f"Usuario: {incoming_msg}\n"
                f"Asistente:"
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
            respuesta_generada = tokenizer.decode(outputs[0], skip_special_tokens=True)
            respuesta = respuesta_generada.replace(prompt, "").strip()

        print("Respuesta generada:", respuesta)
        return f"<Response><Message>{respuesta}</Message></Response>"

    except Exception as e:
        print(f"Error al procesar el mensaje: {e}")
        return "<Response><Message>Lo siento, hubo un error al procesar tu solicitud.</Message></Response>"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)