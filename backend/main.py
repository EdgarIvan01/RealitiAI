"""
RealitiAI — Backend FastAPI
Asistente emocional para problemas de pareja/amistad con IA
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
import os
import json
import re
import asyncio
import tempfile
import base64
from dotenv import load_dotenv
import io
import matplotlib
matplotlib.use('Agg') # Obligatorio para servidores sin interfaz gráfica
import matplotlib.pyplot as plt
from transformers import pipeline
load_dotenv()

app = FastAPI(title="RealitiAI Backend", version="1.0.0")

# ══════════════════════════════════════
#  CORS
# ══════════════════════════════════════
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════
#  CLIENTES OpenAI / DeepSeek
# ══════════════════════════════════════

# Async para streaming de chat
openai_async = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30.0,
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30.0,
)

deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    timeout=60.0,
)

# ══════════════════════════════════════
#  PROMPT BASE
# ══════════════════════════════════════
SYSTEM_PROMPT = """Eres un psicólogo directo y honesto. No complaces. Si el usuario tiene la culpa, se lo dices. Si la relación es tóxica, lo dices. Respuestas naturales, a veces crudas, sin falsa empatía.

Reglas:
- Sé directo, no uses rodeos
- Si hay manipulación, señálala claramente
- Si el usuario se autoengaña, dile
- Usa ejemplos concretos cuando puedas
- No minimices el dolor, pero tampoco exageres
- Si hay riesgo, sé claro en la alerta
- Puedes usar lenguaje coloquial si la situación lo amerita
- Responde en español
- Formato: puedes usar markdown para listar puntos o destacar cosas importantes
- No digas "como terapeuta" ni "como psicólogo", simplemente responde"""

# ══════════════════════════════════════
#  MODELOS HUGGINGFACE (Carga al inicio)
# ══════════════════════════════════════
sentiment_pipe = None
toxicity_pipe = None
sarcasm_pipe = None
bert_pipe = None

@app.on_event("startup")
async def load_ml_models():
    global sentiment_pipe, toxicity_pipe, sarcasm_pipe, bert_pipe
    try:
        print("Cargando modelos de HuggingFace (puede tardar la primera vez)...")
        sentiment_pipe = pipeline(
            "text-classification",
            model="UMUTeam/roberta-spanish-sentiment-analysis",
            top_k=1
        )
        toxicity_pipe = pipeline("text-classification", model="gplsi/Toxicity_model_RoBERTa-base-bne", top_k=None)
        sarcasm_pipe = pipeline("text-classification", model="l52mas/ironiaL52_roberta", top_k=None)
        print("✅ Modelos base cargados correctamente.")
    except Exception as e:
        print(f"❌ Error cargando modelos base: {e}")

    # ── Modelo BERT personalizado (850 frases) ──
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer as BertTokenizer
        _bert_tokenizer = BertTokenizer.from_pretrained('./tokenizer_bert_850_frases')
        _bert_model = AutoModelForSequenceClassification.from_pretrained('./modelo_bert_850_frases')
        bert_pipe = pipeline("text-classification", model=_bert_model, tokenizer=_bert_tokenizer, top_k=1)
        if hasattr(_bert_model.config, 'id2label'):
            print(f"   BERT labels: {_bert_model.config.id2label}")
        print("✅ Modelo BERT personalizado cargado correctamente.")
    except Exception as e:
        print(f"⚠️ Modelo BERT no encontrado (análisis será sin BERT): {e}")
        bert_pipe = None

# ══════════════════════════════════════
#  MODELOS Pydantic
# ══════════════════════════════════════
class ChatRequest(BaseModel):
    text: str
    historial: list = []

class TTSRequest(BaseModel):
    text: str
    voice: str = "nova"
    speed: float = 1.0

class DecisionRequest(BaseModel):
    full_conversation: list
    cuestionario: dict = {}

class DetectRelacionRequest(BaseModel):
    text: str


# ══════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════

@app.get("/health")
async def health():
    """Verifica que el backend está corriendo."""
    return {"status": "ok", "service": "realiti-ai-backend"}


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Chat principal con streaming usando GPT-4o-mini.
    El frontend envía { text, historial }.
    historial ya incluye el mensaje actual del usuario.
    """
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        historial = list(req.historial)

        # Reemplazar último mensaje de usuario con req.text
        # (para imágenes, req.text contiene la descripción completa de visión)
        if historial and historial[-1].get("role") == "user" and req.text:
            historial[-1]["content"] = req.text
        elif req.text:
            historial.append({"role": "user", "content": req.text})

        # Limitar historial a últimos 20 mensajes para ahorrar tokens
        historial_limitado = historial[-20:] if len(historial) > 20 else historial
        for msg in historial_limitado:
            if msg.get("role") in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        async def generate():
            stream = await openai_async.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                stream=True,
                temperature=0.8,
                max_tokens=1500,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        return StreamingResponse(
            generate(),
            media_type="text/plain; charset=utf-8"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en chat: {str(e)}")


@app.post("/analyze-image")
async def analyze_image(
    image: UploadFile = File(...),
    question: str = Form("")
):
    """Analiza imagen con visión de GPT-4o-mini y devuelve descripción."""
    try:
        image_bytes = await image.read()
        base64_image = base64.b64encode(image_bytes).decode()
        mime_type = image.content_type or "image/jpeg"

        vision_prompt = """Analiza esta captura de pantalla de un chat de pareja.
Tu tarea:
1. TRANSCRIBIR todo el texto visible en la imagen lo más fielmente posible
2. Si no puedes leer algo, pon [ilegible]
3. NO inventes texto que no esté en la imagen
4. Si la imagen está borrosa, muy pequeña o protegida, responde exactamente: DETECCIÓN_FALLIDA
5. Si no hay texto en la imagen, responde exactamente: DESCRIPCIÓN_VACÍA"""

        user_content = [{"type": "text", "text": vision_prompt}]

        if question.strip():
            user_content.append({
                "type": "text",
                "text": f"\n\nNota adicional del usuario: {question}"
            })

        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
        })

        def _call_vision():
            return openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": user_content}],
                max_tokens=2000,
            )

        response = await asyncio.to_thread(_call_vision)
        description = response.choices[0].message.content.strip()

        if not description:
            description = "DESCRIPCIÓN_VACÍA"

        return {"description": description}

    except Exception as e:
        return {"description": f"ERROR_{str(e)[:80]}"}


@app.post("/transcribe-audio")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio con Whisper de OpenAI."""
    try:
        audio_bytes = await audio.read()

        # Determinar extensión
        ext = ".webm"
        if audio.content_type:
            if "mp3" in audio.content_type:
                ext = ".mp3"
            elif "wav" in audio.content_type:
                ext = ".wav"
            elif "m4a" in audio.content_type or "mp4" in audio.content_type:
                ext = ".m4a"

        # Escribir a archivo temporal
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            def _transcribe():
                with open(temp_path, "rb") as f:
                    return openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        language="es",
                    )

            transcript = await asyncio.to_thread(_transcribe)
            return {"text": transcript.text}

        finally:
            os.unlink(temp_path)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en transcripción: {str(e)}"
        )


@app.post("/text-to-speech")
async def text_to_speech(req: TTSRequest):
    """Genera audio con TTS de OpenAI (tts-1-hd)."""
    try:
        # Limpiar markdown del texto
        clean_text = req.text
        clean_text = re.sub(r'<[^>]+>', '', clean_text)
        clean_text = clean_text.replace("**", "").replace("##", "")
        clean_text = clean_text.replace("#", "").replace("*", "")
        clean_text = clean_text.replace("_", "").replace("`", "")
        clean_text = clean_text.replace("|", "").replace("---", "")

        # Límite de TTS (4096 caracteres)
        if len(clean_text) > 4000:
            clean_text = clean_text[:4000] + "..."

        def _generate_audio():
            response = openai_client.audio.speech.create(
                model="tts-1-hd",
                voice=req.voice,
                input=clean_text,
                speed=req.speed,
            )
            return b"".join(response.iter_bytes())

        audio_bytes = await asyncio.to_thread(_generate_audio)

        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en TTS: {str(e)}"
        )


@app.post("/decision")
async def decision(req: DecisionRequest):
    """
    Análisis final con DeepSeek.
    Recibe la conversación completa y el cuestionario.
    Devuelve decisión tajante con categoría.
    """
    try:
        # Formatear conversación
        conv_text = ""
        for msg in req.full_conversation:
            role = "Usuario" if msg.get("role") == "user" else "Asistente"
            content = msg.get("content", "")
            # Limpiar markdown del contenido
            content = re.sub(r'<[^>]+>', '', content)
            content = content.replace("**", "").replace("*", "")
            if content.strip():
                conv_text += f"{role}: {content}\n\n"

        # Formatear cuestionario
        labels = {
            "tiempo_conocerse": "¿Cuánto tiempo llevan de conocerse?",
            "tiempo_pareja": "¿Cuánto tiempo como pareja?",
            "hablar_emociones": "¿Han intentado hablar de emociones?",
            "violencia_fisica": "¿Hay violencia física?",
            "violencia_psicologica": "¿Hay violencia psicológica?",
        }
        cuest_text = ""
        if req.cuestionario:
            for key, value in req.cuestionario.items():
                label = labels.get(key, key)
                cuest_text += f"- {label}: {value}\n"

        # Alerta de violencia
        violence_alert = ""
        if req.cuestionario:
            vf = req.cuestionario.get("violencia_fisica", "").lower()
            vp = req.cuestionario.get("violencia_psicologica", "").lower()
            if "sí" in vf or "si" in vf:
                violence_alert = "\n⚠️ ALERTA CRÍTICA: El usuario reportó violencia física. Prioriza su seguridad en la respuesta.\n"
            elif "sí" in vp or "si" in vp:
                violence_alert = "\n⚠️ ALERTA: El usuario reportó violencia psicológica. Tomalo en cuenta.\n"

        prompt = f"""Basado en esta conversación y el cuestionario, da una conclusión corta y tajante.
Clasifica en: 'Terminar definitivamente', 'Es salvable con terapia', 'Estás exagerando, todo está en tu cabeza'.
Máximo 3 párrafos. Sin rodeos. Sé brutalmente honesto.
{violence_alert}
CONVERSACIÓN:
{conv_text if conv_text.strip() else '(Sin conversación previa)'}

CUESTIONARIO:
{cuest_text if cuest_text.strip() else '(No respondido)'}

IMPORTANTE: Al final de tu respuesta, en una línea aparte, incluye exactamente:
CATEGORÍA: [terminar|terapia|exageracion]"""

        def _call_deepseek():
            return deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un psicólogo directo y honesto que da conclusiones tajantes sobre relaciones de pareja y amistad. No tienes filtro."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800,
            )

        response = await asyncio.to_thread(_call_deepseek)
        text = response.choices[0].message.content

        # Extraer categoría
        category = "exageracion"
        match = re.search(
            r'CATEGORÍA:\s*(terminar|terapia|exageracion)',
            text,
            re.IGNORECASE
        )
        if match:
            category = match.group(1).lower()

        # Limpiar texto (quitar la línea de categoría)
        clean_text = re.sub(r'CATEGORÍA:\s*\w+', '', text).strip()
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)

        return {"decision": clean_text, "category": category}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en decisión: {str(e)}"
        )


@app.post("/detect_relacion")
async def detect_relacion(req: DetectRelacionRequest):
    """
    Detecta si el mensaje habla de relación de pareja o amistad profunda.
    Primero usa keywords (rápido), luego GPT-4o-mini si es ambiguo.
    """
    try:
        text_lower = req.text.lower()

        # ── Fase 1: Keywords (instantáneo) ──
        pareja_keywords = [
            'novio', 'novia', 'pareja', 'esposo', 'esposa', 'matrimonio',
            'boda', 'casarse', 'divorcio', 'separar', 'separación',
            'mi ex', 'ex novio', 'ex novia', 'ex pareja',
            'relación', 'relacion', 'relaciones',
            'enamorado', 'enamorada', 'amor', 'te amo', 'te quiero',
            'infidelidad', 'engaño', 'traición', 'traicion', 'traiciono',
            'celoso', 'celosa', 'celos',
            'mi niño', 'mi niña', 'mi viejo', 'mi vieja',
            'noviazgo', 'compromiso', 'comprometido', 'comprometida',
            'citas', 'salir con', 'andar con',
            'me dejó', 'me dejo', 'lo dejé', 'la dejé',
            'no me escribe', 'no me contesta', 'me ignora',
            'lo amo', 'la amo', 'lo quiero', 'la quiero',
        ]

        amistad_keywords = [
            'amigo', 'amiga', 'amistad', 'mejor amigo', 'mejor amiga',
            'compañero', 'compañera', 'cuates', 'panas',
            'amigos desde', 'conocí a',
        ]

        pareja_hits = sum(1 for k in pareja_keywords if k in text_lower)
        amistad_hits = sum(1 for k in amistad_keywords if k in text_lower)
        total_hits = pareja_hits + amistad_hits

        if total_hits >= 2:
            confidence = min(0.95, 0.65 + total_hits * 0.08)
            return {"is_relationship": True, "confidence": round(confidence, 2)}

        if total_hits == 1:
            return {"is_relationship": True, "confidence": 0.6}

        # ── Fase 2: GPT-4o-mini (para casos ambiguos) ──
        def _call_detect():
            return openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Determina si el siguiente mensaje habla de una relación "
                            "de pareja (noviazgo, matrimonio, ex, problemas románticos) "
                            "o amistad profunda con problemas.\n"
                            'Responde SOLO con JSON válido sin markdown ni backticks:\n'
                            '{"is_relationship": true, "confidence": 0.8}'
                        )
                    },
                    {"role": "user", "content": req.text[:500]}
                ],
                temperature=0,
                max_tokens=50,
            )

        response = await asyncio.to_thread(_call_detect)
        result_text = response.choices[0].message.content.strip()
        result_text = result_text.replace("```json", "").replace("```", "").strip()

        try:
            result = json.loads(result_text)
            return {
                "is_relationship": bool(result.get("is_relationship", False)),
                "confidence": round(float(result.get("confidence", 0.0)), 2)
            }
        except json.JSONDecodeError:
            return {"is_relationship": False, "confidence": 0.0}

    except Exception as e:
        # En caso de error, no bloquear al usuario
        return {"is_relationship": False, "confidence": 0.0}


# ══════════════════════════════════════
#  SCHEMA PARA ANÁLISIS DUAL
# ══════════════════════════════════════
class AnalyzeDualRequest(BaseModel):
    conversation: list

# ══════════════════════════════════════
#  ENDPOINT /analyze/dual
# ══════════════════════════════════════
@app.post("/analyze/dual")
async def analyze_dual(req: AnalyzeDualRequest):
    """
    Análisis híbrido de conversación:
    PASO 1 → Detectar si hay mensajes de la pareja
    PASO 2A → Análisis psicológico con IA (siempre)
    PASO 2B → Extraer mensajes de pareja con IA (si hay)
    PASO 2C → Evaluar con modelos locales: Sentimiento + BERT (si hay)
    PASO 2D → Generar gráficas
    """
    try:
        conv_text = "\n".join([
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in req.conversation
        ])

        # Seguridad: conversación vacía
        if not conv_text.strip():
            return {
                "analysis_type": "general_only",
                "general_analysis": {
                    "sentimiento": "neutral", "patrones": [], "sesgos": [],
                    "nivel_conflicto": "bajo", "riesgo": "bajo",
                    "recomendacion": "No hay conversación para analizar.",
                    "analisis_completo": ""
                },
                "partner_analysis": None
            }

        # Truncar para la IA si es muy largo
        conv_for_ai = conv_text[:8000] if len(conv_text) > 8000 else conv_text

        # ═══════════════════════════════════════
        #  PASO 1: Detectar si hay mensajes de pareja
        # ═══════════════════════════════════════
        def _detect_partner():
            res = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Determina si en esta conversación el usuario incluyó mensajes, "
                            "citas o palabras textuales de su pareja o amigo.\n\n"
                            "Indicadores de SÍ:\n"
                            '- "ella dijo...", "me contestó...", "me escribió..."\n'
                            "- Citas entre comillas atribuidas a la otra persona\n"
                            "- Transcripciones de chats de WhatsApp, Messenger, etc.\n"
                            "- Frases como: lo que me dijo, su respuesta fue, me mandó esto\n\n"
                            "Indicadores de NO:\n"
                            "- El usuario solo cuenta su versión sin citar textualmente a la otra persona\n"
                            "- No hay citas ni mensajes directos del otro\n\n"
                            "Responde SOLO: SI o NO"
                        )
                    },
                    {"role": "user", "content": conv_for_ai[:3000]}
                ],
                temperature=0,
                max_tokens=5,
            )
            return "SI" in res.choices[0].message.content.upper()

        has_partner = await asyncio.to_thread(_detect_partner)

        # ═══════════════════════════════════════
        #  PASO 2A: Análisis General con IA (SIEMPRE)
        # ═══════════════════════════════════════
        def _general_analysis():
            contexto = (
                "considerando AMBOS lados (lo que dice el usuario Y las citas textuales de su pareja/amigo)"
                if has_partner
                else "basándote solo en lo que cuenta el usuario (no hay citas de la otra persona)"
            )
            res = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Eres un psicólogo experto en relaciones interpersonales. "
                            f"Genera un análisis psicológico profundo {contexto}.\n\n"
                            "Devuelve SOLO JSON puro sin markdown ni backticks:\n"
                            "{\n"
                            '  "sentimiento": "positivo|neutral|negativo",\n'
                            '  "patrones": ["patrón emocional 1", "patrón 2"],\n'
                            '  "sesgos": ["sesgo cognitivo 1", "sesgo 2"],\n'
                            '  "nivel_conflicto": "bajo|medio|alto",\n'
                            '  "riesgo": "bajo|medio|alto",\n'
                            '  "recomendacion": "recomendación concreta y directa",\n'
                            '  "analisis_completo": "análisis extenso y detallado"\n'
                            "}"
                        )
                    },
                    {"role": "user", "content": f"Conversación:\n{conv_for_ai}"}
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            text = res.choices[0].message.content
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)

        # ═══════════════════════════════════════
        #  PASO 2B: Extraer mensajes de pareja con IA
        # ═══════════════════════════════════════
        def _extract_partner_messages():
            res = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extrae TODOS los mensajes, citas o frases que la pareja/amigo "
                            "dijo según esta conversación.\n\n"
                            "Incluye:\n"
                            "- Mensajes entre comillas\n"
                            '- "Ella dijo: ...", "Me contestó: ...", "Me escribió: ..."\n'
                            "- Transcripciones de chat de WhatsApp\n"
                            "- Cualquier palabra textual atribuida a la otra persona\n\n"
                            "Devuelve SOLO un array JSON de strings sin markdown ni backticks:\n"
                            '["frase completa 1", "frase completa 2"]\n'
                            "Si no hay mensajes de la pareja, devuelve: []"
                        )
                    },
                    {"role": "user", "content": f"Conversación:\n{conv_for_ai}"}
                ],
                temperature=0,
                max_tokens=2000,
            )
            text = res.choices[0].message.content
            text = text.replace("```json", "").replace("```", "").strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                match = re.search(r'\[.*\]', text, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group())
                    except Exception:
                        return []
                return []

        # ═══════════════════════════════════════
        #  Ejecutar: general + extracción en PARALELO si hay pareja
        # ═══════════════════════════════════════
        if has_partner:
            general_res, partner_msgs = await asyncio.gather(
                asyncio.to_thread(_general_analysis),
                asyncio.to_thread(_extract_partner_messages),
            )
        else:
            general_res = await asyncio.to_thread(_general_analysis)
            partner_msgs = []

        # Caso sin pareja: devolver solo general
        if not has_partner:
            return {
                "analysis_type": "general_only",
                "general_analysis": general_res,
                "partner_analysis": None,
            }

        # ═══════════════════════════════════════
        #  PASO 2C: Evaluar con modelos locales
        # ═══════════════════════════════════════
        partner_analysis = {
            "messages": partner_msgs,
            "sentiment_results": [],
            "bert_results": [],
            "sentiment_summary": {},
            "bert_summary": {},
            "charts": "",
        }

        if partner_msgs and sentiment_pipe:
            sentiment_counts = {}
            bert_counts = {}

            for msg in partner_msgs[:15]:  # Limitar a 15 por rendimiento
                if not msg or not msg.strip():
                    continue
                try:
                    # ── Sentimiento ──
                    s_out = sentiment_pipe(msg)
                    if s_out and len(s_out) > 0:
                        s_label = s_out[0]["label"]
                        s_score = round(s_out[0]["score"], 3)
                        sentiment_counts[s_label] = sentiment_counts.get(s_label, 0) + 1
                        partner_analysis["sentiment_results"].append({
                            "mensaje": msg[:60] + ("..." if len(msg) > 60 else ""),
                            "etiqueta": s_label,
                            "confianza": s_score,
                        })

                    # ── BERT personalizado ──
                    if bert_pipe:
                        b_out = bert_pipe(msg)
                        if b_out and len(b_out) > 0:
                            b_label = b_out[0]["label"]
                            b_score = round(b_out[0]["score"], 3)
                            bert_counts[b_label] = bert_counts.get(b_label, 0) + 1
                            partner_analysis["bert_results"].append({
                                "mensaje": msg[:60] + ("..." if len(msg) > 60 else ""),
                                "etiqueta": b_label,
                                "confianza": b_score,
                            })
                except Exception as e:
                    print(f"Error evaluando mensaje con modelos: {e}")

            partner_analysis["sentiment_summary"] = sentiment_counts
            partner_analysis["bert_summary"] = bert_counts

            # ═══════════════════════════════════════
            #  PASO 2D: Generar Gráficas
            # ═══════════════════════════════════════
            try:
                has_bert_data = bert_pipe and bert_counts
                n_plots = 2 if has_bert_data else 1
                fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4.5))
                if n_plots == 1:
                    axes = [axes]

                # Mapeo de colores para sentimiento
                sentiment_colors = {
                    "Positivo": "#22c55e", "Positive": "#22c55e",
                    "positive": "#22c55e", "positivo": "#22c55e",
                    "Neutral": "#eab308", "neutral": "#eab308",
                    "Negativo": "#ef4444", "Negative": "#ef4444",
                    "negative": "#ef4444", "negativo": "#ef4444",
                }

                # Mapeo de colores para BERT
                bert_colors = {
                    "tóxico": "#ef4444", "toxico": "#ef4444", "toxic": "#ef4444",
                    "manipulador": "#f97316", "manipuladora": "#f97316",
                    "manipulacion": "#f97316", "manipulación": "#f97316",
                    "negativo": "#f59e0b", "negativa": "#f59e0b",
                    "neutro": "#3b82f6", "neutral": "#3b82f6",
                    "positivo": "#22c55e", "positiva": "#22c55e",
                    "normal": "#06b6d4", "cariñoso": "#ec4899",
                    "cariñosa": "#ec4899", "agresivo": "#dc2626",
                    "agresiva": "#dc2626", "pasivo_agresivo": "#a855f7",
                    "controlador": "#ea580c", "controladora": "#ea580c",
                    "inseguro": "#f472b6", "insegura": "#f472b6",
                    "culpabilizador": "#b91c1c", "victimista": "#9333ea",
                }
                default_color = "#8b5cf6"

                # ── Gráfica 1: Sentimiento ──
                if sentiment_counts:
                    labels = list(sentiment_counts.keys())
                    values = list(sentiment_counts.values())
                    colors = [sentiment_colors.get(l, default_color) for l in labels]
                    bars = axes[0].bar(
                        labels, values, color=colors,
                        edgecolor="white", linewidth=1.2, width=0.55
                    )
                    axes[0].set_title(
                        "Sentimiento en mensajes\nde la pareja",
                        fontweight="bold", fontsize=12, color="white", pad=12
                    )
                    axes[0].set_ylabel("Cantidad", color="white", fontsize=10)
                    for bar, v in zip(bars, values):
                        axes[0].text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.15,
                            str(v), ha="center", fontweight="bold",
                            color="white", fontsize=11
                        )

                # ── Gráfica 2: BERT ──
                if has_bert_data:
                    labels_b = list(bert_counts.keys())
                    values_b = list(bert_counts.values())
                    colors_b = [bert_colors.get(l.lower(), default_color) for l in labels_b]
                    bars_b = axes[1].barh(
                        labels_b, values_b, color=colors_b,
                        edgecolor="white", linewidth=1.2, height=0.5
                    )
                    axes[1].set_title(
                        "Clasificación BERT\n(modelo personalizado)",
                        fontweight="bold", fontsize=12, color="white", pad=12
                    )
                    axes[1].set_xlabel("Cantidad", color="white", fontsize=10)
                    for bar, v in zip(bars_b, values_b):
                        axes[1].text(
                            bar.get_width() + 0.15,
                            bar.get_y() + bar.get_height() / 2,
                            str(v), va="center", fontweight="bold",
                            color="white", fontsize=11
                        )

                # Estilo dark theme
                fig.patch.set_facecolor("#0f172a")
                for ax in axes:
                    ax.set_facecolor("#1e293b")
                    ax.tick_params(colors="white", labelsize=9)
                    ax.xaxis.label.set_color("white")
                    ax.yaxis.label.set_color("white")
                    for spine in ax.spines.values():
                        spine.set_color("#334155")
                    ax.grid(axis="y", alpha=0.12, color="#475569")

                plt.tight_layout(pad=2.0)
                buf = io.BytesIO()
                plt.savefig(
                    buf, format="png", bbox_inches="tight",
                    dpi=130, facecolor=fig.get_facecolor()
                )
                plt.close(fig)
                partner_analysis["charts"] = base64.b64encode(
                    buf.getvalue()
                ).decode("utf-8")

            except Exception as e:
                print(f"Error generando gráficas: {e}")

        return {
            "analysis_type": "general_and_models",
            "general_analysis": general_res,
            "partner_analysis": partner_analysis,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en análisis dual: {str(e)}"
        )