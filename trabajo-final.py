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
   
def connectDatabase():
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate("db.json")
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
            "Eres un asistente virtual de una tienda de electronicos ubicada en avenida wilson del centro de lima, una tienda especializada en electronicos de entretenimiento. "
            "Estás capacitado para responder preguntas sobre electronicos de entretenimiento como consola de videojuegos, televisores, sistemas de audio y laptops. "
            "Responde siempre de manera amable, clara y profesional."
        )

    def normalizar(texto):
        texto = texto
        texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
        return texto

    def extraer_electro(mensaje):
        mensaje_norm = normalizar(mensaje)
        electro_ref = db.collection("Gamer").stream()

        for doc in electro_ref:
            nombre_electro = doc.to_dict().get("Nombre")
            nombre_norm = normalizar(nombre_electro)

            if nombre_norm in mensaje_norm:
                return doc.Nombre
        return None

    def buscar_precio(plato):
        ref = db.collection("Gamer").document(plato)
        doc = ref.get()

        if doc.exists:
            data = doc.to_dict()
            producto = data.get("Nombre")
            precio = data.get("Precio")

            return f"Claro, el {producto} cuesta S/{precio}."
        else:
            return None

    return obtener_contexto, extraer_electro, buscar_precio
    
def init():
    model, tokenizer = load_model("carloshuamani/Llama-3.2-ArticulosElectronicos")
    db = connectDatabase()
    twilio_client = initTwilioClient()  # Inicializar el cliente de Twilio aquí
    return model, tokenizer, db, twilio_client # Devolver el cliente

model, tokenizer, db, twilio_client = init() # Obtener el cliente

app = Flask(__name__)


@app.route('/whatsapp', methods=["GET", "POST"])
def whatsapp_mymessage():
    (
        obtener_contexto,
        extraer_electro,
        buscar_precio,
    ) = utils(db)
    print("Request recibido:", request)
    incoming_msg = request.values.get('Body', '')
    print("Mensaje recibido:", incoming_msg)

    respuesta = ""
    contexto = obtener_contexto()

    precio_keywords = ["precio", "cuánto cuesta", "cual es el costo", "cuanto vale"]

    try:
        if any(keyword in incoming_msg for keyword in precio_keywords):
            producto = extraer_electro(incoming_msg)
            if producto:
                respuesta = buscar_precio(producto)
                if not respuesta:
                    respuesta = f"Lo siento, no encontré detalles para el producto '{producto}'."
            else:
                respuesta = "No contamos con ese prodcuto ¿Puedes pedir otro?"
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)