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
    allow_origins=["*"],
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
bert_pipe = None

@app.on_event("startup")
async def load_ml_models():
    #global sentiment_pipe  # Solo mantenemos este
    #try:
    #    print("Cargando modelo de análisis de sentimiento (UMUTeam/roberta-spanish-sentiment-analysis)...")
        #sentiment_pipe = pipeline(
        #    "text-classification",
        #    model="UMUTeam/roberta-spanish-sentiment-analysis",
        #    top_k=1
        #)
    #    print("Modelo de sentimiento desactivado.")
    #    sentiment_pipe = None
    #except Exception as e:
    #    print(f"❌ Modelo de sentimiento desactivado: {e}")
    #    sentiment_pipe = None

    # ── Modelo BERT personalizado (850 frases) ──
    #try:
    #    from transformers import AutoModelForSequenceClassification, AutoTokenizer as BertTokenizer
        # _bert_tokenizer = BertTokenizer.from_pretrained('./tokenizer_bert_850_frases') 
        # _bert_model = AutoModelForSequenceClassification.from_pretrained('./modelo_bert_850_frases')
        # bert_pipe = pipeline("text-classification", model=_bert_model, tokenizer=_bert_tokenizer, top_k=1)
        bert_pipe = None
        #if hasattr(_bert_model.config, 'id2label'):
        #    print(f"   BERT labels: {_bert_model.config.id2label}")
    #    print("Modelo BERT desactivado para deploy.")
    #except Exception as e:
    #    print(f"Error con BERT: {e}")
    #    bert_pipe = None

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
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        historial = list(req.historial)

        if historial and historial[-1].get("role") == "user" and req.text:
            historial[-1]["content"] = req.text
        elif req.text:
            historial.append({"role": "user", "content": req.text})

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
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",      # ← CRÍTICO para Render
                "Connection": "keep-alive",
            }
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
4. Si la imagen está borrosa, muy pequeña o protegida, responde exactamente: No puedo reconocer nada
5. Si no hay texto en la imagen, responde con una descripcion no tan detallada ejemplo: es un gatito"""

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
    Análisis híbrido usando SOLO OpenAI (sin modelos locales).
    Compatible con Render free tier.
    """
    try:
        conv_text = "\n".join([
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in req.conversation
        ])

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

        conv_for_ai = conv_text[:8000]

        # Detectar si hay mensajes de pareja
        def _detect_partner():
            res = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Determina si el usuario incluyó mensajes/citas textuales de su pareja.\n"
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

        # Análisis general (siempre)
        def _general_analysis():
            contexto = (
                "considerando AMBOS lados (usuario Y citas de su pareja)"
                if has_partner
                else "basándote solo en lo que cuenta el usuario"
            )
            res = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Eres un psicólogo experto. Genera un análisis {contexto}.\n"
                            "Devuelve SOLO JSON puro sin markdown:\n"
                            '{"sentimiento":"positivo|neutral|negativo","patrones":["p1"],"sesgos":["s1"],'
                            '"nivel_conflicto":"bajo|medio|alto","riesgo":"bajo|medio|alto",'
                            '"recomendacion":"rec","analisis_completo":"texto largo"}'
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

        # Extraer mensajes de pareja + análisis con GPT (sin modelos locales)
        def _extract_and_analyze_partner():
            res = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extrae TODOS los mensajes/citas de la pareja y clasifica cada uno.\n\n"
                            "Devuelve SOLO JSON puro sin markdown:\n"
                            '{"messages":["frase1","frase2"],'
                            '"sentiment_results":[{"mensaje":"...","etiqueta":"Positivo|Neutral|Negativo","confianza":0.85}],'
                            '"sentiment_summary":{"Positivo":1,"Negativo":2,"Neutral":1}}'
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
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group())
                    except:
                        pass
                return {"messages": [], "sentiment_results": [], "sentiment_summary": {}}

        if has_partner:
            general_res, partner_data = await asyncio.gather(
                asyncio.to_thread(_general_analysis),
                asyncio.to_thread(_extract_and_analyze_partner),
            )

            # Generar gráfica simple con matplotlib
            charts_b64 = ""
            sentiment_summary = partner_data.get("sentiment_summary", {})
            if sentiment_summary:
                try:
                    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
                    sentiment_colors = {
                        "Positivo": "#22c55e", "Positive": "#22c55e", "positivo": "#22c55e",
                        "Neutral": "#eab308", "neutral": "#eab308",
                        "Negativo": "#ef4444", "Negative": "#ef4444", "negativo": "#ef4444",
                    }
                    labels = list(sentiment_summary.keys())
                    values = list(sentiment_summary.values())
                    colors = [sentiment_colors.get(l, "#8b5cf6") for l in labels]
                    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2, width=0.55)
                    ax.set_title("Sentimiento en mensajes\nde la pareja", fontweight="bold", fontsize=12, color="white", pad=12)
                    ax.set_ylabel("Cantidad", color="white", fontsize=10)
                    for bar, v in zip(bars, values):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, str(v),
                                ha="center", fontweight="bold", color="white", fontsize=11)
                    fig.patch.set_facecolor("#0f172a")
                    ax.set_facecolor("#1e293b")
                    ax.tick_params(colors="white", labelsize=9)
                    for spine in ax.spines.values():
                        spine.set_color("#334155")
                    ax.grid(axis="y", alpha=0.12, color="#475569")
                    plt.tight_layout(pad=2.0)
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor=fig.get_facecolor())
                    plt.close(fig)
                    charts_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                except Exception as e:
                    print(f"Error gráfica: {e}")

            partner_analysis = {
                "messages": partner_data.get("messages", []),
                "sentiment_results": partner_data.get("sentiment_results", []),
                "bert_results": [],  # Sin modelo local
                "sentiment_summary": sentiment_summary,
                "bert_summary": {},
                "charts": charts_b64,
            }

            return {
                "analysis_type": "general_and_models",
                "general_analysis": general_res,
                "partner_analysis": partner_analysis,
            }
        else:
            general_res = await asyncio.to_thread(_general_analysis)
            return {
                "analysis_type": "general_only",
                "general_analysis": general_res,
                "partner_analysis": None,
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis: {str(e)}")