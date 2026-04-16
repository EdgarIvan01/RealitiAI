import { Component, ChangeDetectorRef, ElementRef, ViewChild, AfterViewChecked, NgZone, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { HttpClientModule, HttpClient } from '@angular/common/http';
import { marked } from 'marked';

interface Mensaje {
  role: 'user' | 'bot';
  text: string;
  raw?: string;
  imageUrl?: string;
  isImage?: boolean;
  isAudio?: boolean;
  audioUrl?: string;
}

interface Chat {
  id: number;
  title: string;
  messages: Mensaje[];
  fecha: Date;
}

const API_BASE = 'https://trickle-matriarch-number.ngrok-free.dev';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule, HttpClientModule],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css']
})
export class ChatComponent implements AfterViewChecked, OnDestroy, OnInit {

  @ViewChild('messageContainer') private messageContainer!: ElementRef;
  @ViewChild('chatInput') private chatInputRef!: ElementRef;

  input: string = '';
  messages: Mensaje[] = [];
  chats: Chat[] = [];
  chatActual: number | null = null;
  historialCompleto: any[] = [];
  isTyping: boolean = false;
  mensajeBienvenida: string = '';
  imagenSeleccionada: string | null = null;
  imagenFile: File | null = null;
  imagenGrande: string | null = null;
  mediaRecorder: any = null;
  audioChunks: Blob[] = [];
  isRecording: boolean = false;
  recordingTime: number = 0;
  recordingInterval: any = null;
  audioUrl: string | null = null;
  audioBlob: Blob | null = null;
  mostrarReproduccion: boolean = false;
  selectedVoice: string = 'nova';
  speechRate: number = 1.0;
  autoPlayTTS: boolean = false;
  currentAudio: HTMLAudioElement | null = null;
  currentAudioIndex: number | null = null;
  isAudioPlaying: boolean = false;
  isAudioPaused: boolean = false;
  isTTSLoading: boolean = false;
  showSidebar: boolean = false;
  showCuestionario: boolean = false;
  cuestionarioRespondido: boolean = false;
  cuestionarioRespuestas: any = { tiempo_conocerse: '', tiempo_pareja: '', hablar_emociones: '', violencia_fisica: '', violencia_psicologica: '' };
  mensajePendiente: string | null = null;
  isDetecting: boolean = false;
  decisionTexto: string = '';
  decisionCategoria: string = '';
  showDecision: boolean = false;
  isDecisionLoading: boolean = false;
  showMobileActions: boolean = false;
  marked = marked;

  // ── NUEVO: Variables para Análisis Dual ──
  showAnalysisModal: boolean = false;
  isAnalysisLoading: boolean = false;
  analysisResult: any = null;
  showFullAnalysis: boolean = false;

  // ── OPTIMIZACIONES: Throttle y scroll ──
  private lastParseTime = 0;
  private parseThrottleMs = 100;
  private needsScroll = false;
  private backendTimeout = 60000; // 60 segundos para cold starts

  constructor(private cdr: ChangeDetectorRef, private ngZone: NgZone, private http: HttpClient) {
    marked.setOptions({ breaks: true, gfm: true });
    this.cargarChatsDeLocalStorage();
    this.mensajeBienvenida = this.obtenerMensajeAleatorio();
  }

  ngOnInit(): void { this.nuevoChat(); }
  ngOnDestroy(): void { this.stopSpeech(); this.limpiarGrabacion(); }
  
  ngAfterViewChecked(): void {
    if (this.needsScroll) {
      this.scrollToBottom();
      this.needsScroll = false;
    }
  }

  scrollToBottom(): void { 
    try { 
      if (this.messageContainer) 
        this.messageContainer.nativeElement.scrollTop = this.messageContainer.nativeElement.scrollHeight; 
    } catch (err) {} 
  }
  
  private requestScroll(): void {
    this.needsScroll = true;
  }

  parseMarkdown(text: string): string { return marked.parse(text) as string; }

  // ═══ TTS ═══
  getPlainText(htmlText: string): string { const t = document.createElement('div'); t.innerHTML = htmlText; return t.textContent || htmlText; }
  
  async speakMessage(text: string, index: number): Promise<void> {
    try {
      if (this.isAudioPlaying && this.currentAudioIndex === index) return;
      if (this.isAudioPaused && this.currentAudioIndex === index) { this.resumeSpeech(); return; }
      this.stopSpeech();
      const p = this.getPlainText(text); if (!p.trim()) return;
      this.currentAudioIndex = index; this.isTTSLoading = true; this.cdr.detectChanges();
      const res = await fetch(`${API_BASE}/text-to-speech`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: p, voice: this.selectedVoice, speed: this.speechRate }) });
      if (!res.ok) throw new Error();
      const blob = await res.blob(); if (blob.size === 0) throw new Error();
      const url = URL.createObjectURL(blob); this.currentAudio = new Audio(url); this.isAudioPlaying = true; this.isAudioPaused = false; this.isTTSLoading = false;
      this.currentAudio.onended = () => { this.ngZone.run(() => { this.limpiarEstadoAudio(); URL.revokeObjectURL(url); }); };
      this.currentAudio.onerror = () => { this.ngZone.run(() => { this.limpiarEstadoAudio(); URL.revokeObjectURL(url); }); };
      await this.currentAudio.play(); this.cdr.detectChanges();
    } catch (e) { this.ngZone.run(() => this.limpiarEstadoAudio()); }
  }
  
  pauseSpeech(): void { if (this.currentAudio && this.isAudioPlaying) { this.currentAudio.pause(); this.isAudioPaused = true; this.isAudioPlaying = false; } }
  resumeSpeech(): void { if (this.currentAudio && this.isAudioPaused) { this.currentAudio.play(); this.isAudioPaused = false; this.isAudioPlaying = true; } }
  stopSpeech(): void { if (this.currentAudio) { this.currentAudio.pause(); this.currentAudio.currentTime = 0; this.currentAudio.src = ''; this.currentAudio = null; } this.limpiarEstadoAudio(); }
  
  private limpiarEstadoAudio(): void { this.isAudioPlaying = false; this.isAudioPaused = false; this.currentAudioIndex = null; this.isTTSLoading = false; }
  isMessagePlaying(i: number): boolean { return (this.isAudioPlaying || this.isAudioPaused) && this.currentAudioIndex === i; }
  isMessagePaused(i: number): boolean { return this.isAudioPaused && this.currentAudioIndex === i; }
  isTTSLoadingForMessage(i: number): boolean { return this.isTTSLoading && this.currentAudioIndex === i; }
  onVoiceChange(_e: any): void { this.stopSpeech(); }
  onRateChange(e: any): void { this.speechRate = parseFloat(e.target.value); }

  // ═══ ANÁLISIS DUAL ═══
  async analizarPatron(): Promise<void> {
    if (this.historialCompleto.length === 0 || this.isAnalysisLoading) { alert('No hay conversación para analizar.'); return; }
    this.isAnalysisLoading = true; this.showAnalysisModal = true; this.analysisResult = null; this.showFullAnalysis = false; this.cdr.detectChanges();
    try {
      const res = await fetch(`${API_BASE}/analyze/dual`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ conversation: this.historialCompleto }) });
      if (!res.ok) throw new Error('Error del servidor');
      this.analysisResult = await res.json();
    } catch (e) { this.analysisResult = { error: 'No se pudo conectar con el servidor de análisis.' }; }
    finally { this.isAnalysisLoading = false; this.cdr.detectChanges(); }
  }
  
  cerrarAnalysisModal(): void { this.showAnalysisModal = false; this.analysisResult = null; }

  // ═══ DETECCIÓN / CUESTIONARIO / DECISIÓN ═══
  async detectarRelacion(text: string) { try { const r = await fetch(`${API_BASE}/detect_relacion`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) }); return await r.json(); } catch { return { is_relationship: false, confidence: 0 }; } }
  
  enviarCuestionario(): void { if (!this.cuestionarioRespuestas.tiempo_conocerse.trim() || !this.cuestionarioRespuestas.hablar_emociones || !this.cuestionarioRespuestas.violencia_fisica || !this.cuestionarioRespuestas.violencia_psicologica) { alert('Responde los campos obligatorios.'); return; } this.cuestionarioRespondido = true; this.showCuestionario = false; if (this.mensajePendiente) { this.enviarMensajeTexto(this.mensajePendiente); this.mensajePendiente = null; } }
  
  saltarCuestionario(): void { this.showCuestionario = false; if (this.mensajePendiente) { this.enviarMensajeTexto(this.mensajePendiente); this.mensajePendiente = null; } }
  
  cancelarCuestionario(): void { this.showCuestionario = false; if (this.mensajePendiente) { this.input = this.mensajePendiente; this.mensajePendiente = null; } }
  
  async solicitarDecision(): Promise<void> { if (this.historialCompleto.length === 0) { alert('No hay conversación.'); return; } if (this.isDecisionLoading || this.isTyping) return; this.isDecisionLoading = true; this.showDecision = true; this.decisionTexto = ''; this.cdr.detectChanges(); try { const r = await fetch(`${API_BASE}/decision`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ full_conversation: this.historialCompleto, cuestionario: this.cuestionarioRespondido ? this.cuestionarioRespuestas : {} }) }); const d = await r.json(); this.decisionTexto = d.decision; this.decisionCategoria = d.category; } catch { this.decisionTexto = 'Error al generar decisión.'; } finally { this.isDecisionLoading = false; this.cdr.detectChanges(); } }
  
  cerrarDecision(): void { this.showDecision = false; }

  // ═══ CHAT ═══
  obtenerMensajeAleatorio(): string { const m = ["¿De qué quieres hablar?", "¿Qué tienes para contarme?", "¿Qué situación te gustaría hablar?", "¿En qué te gustaría recibir ayuda?"]; return m[Math.floor(Math.random() * m.length)]; }
  
  async enviar(): Promise<void> { if ((!this.input.trim() && !this.imagenFile && !this.audioBlob) || this.isTyping || this.isDetecting) return; if (this.imagenFile) { await this.enviarConImagen(); return; } if (this.audioBlob) { await this.enviarAudio(); return; } const t = this.input.trim(); if (!t) return; if (!this.cuestionarioRespondido) { try { this.isDetecting = true; this.cdr.detectChanges(); const d = await this.detectarRelacion(t); if (d.is_relationship && d.confidence >= 0.6) { this.mensajePendiente = t; this.input = ''; this.showCuestionario = true; return; } } finally { this.isDetecting = false; } } this.enviarMensajeTexto(t); }
  
  enviarMensajeTexto(m: string): void { this.messages.push({ role: 'user', text: m }); this.historialCompleto.push({ role: 'user', content: m }); this.input = ''; this.messages.push({ role: 'bot', text: '', raw: '' }); const i = this.messages.length - 1; this.isTyping = true; this.enviarAlBackend(m, i); }
  
  async enviarConImagen(): Promise<void> { const p = this.input.trim(); const fd = new FormData(); fd.append('image', this.imagenFile!); fd.append('question', p); this.messages.push({ role: 'user', text: p, imageUrl: this.imagenSeleccionada!, isImage: true }); this.historialCompleto.push({ role: 'user', content: p ? `[Imagen y pregunta: ${p}]` : '[Imagen para analizar]' }); this.input = ''; this.imagenSeleccionada = null; this.imagenFile = null; this.messages.push({ role: 'bot', text: '', raw: '' }); const i = this.messages.length - 1; this.isTyping = true; try { const r = await fetch(`${API_BASE}/analyze-image`, { method: 'POST', body: fd }); const d = await r.json(); const desc = d.description; if (desc === 'DETECCIÓN_FALLIDA' || desc.startsWith('ERROR_')) { this.messages[i].text = '⚠️ No pude leer la imagen.'; this.isTyping = false; return; } const msg = p ? `Imagen procesada: ${desc}\nPregunta: ${p}` : `Analiza esta conversación de pareja: ${desc}`; await this.enviarAlBackend(msg, i); } catch { this.messages[i].text = '❌ Error al procesar.'; this.isTyping = false; } }
  
  async enviarAudio(): Promise<void> { if (!this.audioBlob || this.isTyping) return; const ab = this.audioBlob; const au = this.audioUrl; this.messages.push({ role: 'user', text: '', isAudio: true, audioUrl: au! }); const ui = this.messages.length - 1; this.historialCompleto.push({ role: 'user', content: '[Audio]' }); this.audioUrl = null; this.audioBlob = null; this.mostrarReproduccion = false; this.messages.push({ role: 'bot', text: '', raw: '' }); const bi = this.messages.length - 1; this.isTyping = true; try { const fd = new FormData(); fd.append('audio', ab, 'audio.webm'); const r = await fetch(`${API_BASE}/transcribe-audio`, { method: 'POST', body: fd }); const d = await r.json(); const t = d.text || 'No transcrito'; this.messages[ui].text = `"${t}"`; await this.enviarAlBackend(`[AUDIO]: "${t}"`, bi); } catch { this.messages[bi].text = '❌ Error en audio.'; this.isTyping = false; } }
  
  // ═══ ENVIAR AL BACKEND CON OPTIMIZACIONES ═══
  async enviarAlBackend(m: string, i: number): Promise<void> {
    try {
      // Crear controller con timeout para cold starts
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.backendTimeout);

      const r = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: m, historial: this.historialCompleto }),
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (!r.ok) {
        throw new Error(`HTTP ${r.status}`);
      }
      if (!r.body) throw new Error('No streaming support');

      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let full = '';

      const process = ({ done, value }: any): Promise<void> => {
        if (done) {
          this.ngZone.run(() => {
            this.messages[i].raw = full;
            const f = full.trim() || '😊 Entendido.';
            this.messages[i].text = marked.parse(f) as string;
            this.historialCompleto.push({ role: 'assistant', content: f });
            this.isTyping = false;
            this.guardarChatActual();
            this.requestScroll();
            this.cdr.detectChanges();
            if (this.autoPlayTTS) this.speakMessage(f, i);
          });
          return Promise.resolve();
        }
        
        full += decoder.decode(value, { stream: true });
        this.ngZone.run(() => {
          this.messages[i].raw = full;
          // THROTTLE: Solo parsear markdown cada 100ms
          const now = Date.now();
          if (now - this.lastParseTime > this.parseThrottleMs) {
            this.messages[i].text = marked.parse(full + '▌') as string;
            this.lastParseTime = now;
          }
          this.requestScroll();
          this.cdr.detectChanges();
        });
        return reader.read().then(process);
      };
      return reader.read().then(process);
    } catch (e: any) {
      this.ngZone.run(() => {
        if (e.name === 'AbortError') {
          this.messages[i].text = '⏱️ El servidor tardó demasiado. Render puede estar arrancando, intenta de nuevo en unos segundos.';
        } else {
          this.messages[i].text = '😔 Error de conexión. Verifica que el servidor esté activo.';
        }
        this.isTyping = false;
        this.cdr.detectChanges();
      });
    }
  }

  // ═══ multimedia / chat management ═══
  seleccionarImagen(e: any): void { const f = e.target.files[0]; if (!f || !f.type.startsWith('image/')) return; this.imagenFile = f; const r = new FileReader(); r.onload = (ev: any) => this.imagenSeleccionada = ev.target.result; r.readAsDataURL(f); }
  cancelarImagen(): void { this.imagenSeleccionada = null; this.imagenFile = null; }
  abrirImagenGrande(u: string): void { this.imagenGrande = u; }
  cerrarImagenGrande(): void { this.imagenGrande = null; }
  
  async iniciarGrabacion(): Promise<void> { try { const s = await navigator.mediaDevices.getUserMedia({ audio: true }); this.mediaRecorder = new MediaRecorder(s); this.audioChunks = []; this.isRecording = true; this.recordingTime = 0; this.mostrarReproduccion = false; this.recordingInterval = setInterval(() => { this.recordingTime++; }, 1000); this.mediaRecorder.ondataavailable = (e: any) => this.audioChunks.push(e.data); this.mediaRecorder.onstop = () => { this.audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' }); this.audioUrl = URL.createObjectURL(this.audioBlob); this.mostrarReproduccion = true; s.getTracks().forEach((t: any) => t.stop()); }; this.mediaRecorder.start(); } catch { alert('No se pudo acceder al micrófono.'); } }
  
  detenerGrabacion(): void { if (this.mediaRecorder && this.isRecording) { this.mediaRecorder.stop(); this.isRecording = false; clearInterval(this.recordingInterval); } }
  
  cancelarGrabacion(): void { if (this.mediaRecorder && this.isRecording) { this.mediaRecorder.stop(); this.isRecording = false; clearInterval(this.recordingInterval); } this.limpiarGrabacion(); }
  
  private limpiarGrabacion(): void { this.audioBlob = null; this.audioUrl = null; this.mostrarReproduccion = false; this.audioChunks = []; this.recordingTime = 0; if (this.recordingInterval) clearInterval(this.recordingInterval); }
  
  formatearTiempo(s: number): string { return `${Math.floor(s/60).toString().padStart(2,'0')}:${(s%60).toString().padStart(2,'0')}`; }
  toggleSidebar(): void { this.showSidebar = !this.showSidebar; }
  cerrarSidebar(): void { this.showSidebar = false; }
  autoResizeTextarea(t: any): void { t.style.height = 'auto'; t.style.height = Math.min(t.scrollHeight, 150) + 'px'; t.style.overflow = t.scrollHeight > 150 ? 'auto' : 'hidden'; }
  onInputKeydown(e: KeyboardEvent): void { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.enviar(); } }
  
  nuevoChat(): void { if (this.messages.length > 1 && this.chatActual !== null) this.guardarChatActual(); this.stopSpeech(); this.messages = []; this.historialCompleto = []; this.chatActual = null; this.input = ''; this.imagenSeleccionada = null; this.imagenFile = null; this.mensajeBienvenida = this.obtenerMensajeAleatorio(); this.limpiarGrabacion(); this.cuestionarioRespondido = false; this.showCuestionario = false; this.mensajePendiente = null; this.cuestionarioRespuestas = { tiempo_conocerse: '', tiempo_pareja: '', hablar_emociones: '', violencia_fisica: '', violencia_psicologica: '' }; this.showDecision = false; this.showSidebar = false; this.showMobileActions = false; }
  
  guardarChatActual(): void { if (this.messages.length > 1) { const t = this.generarTitulo(); const c: Chat = { id: Date.now(), title: t, messages: [...this.messages], fecha: new Date() }; if (this.chatActual !== null && this.chats[this.chatActual]) this.chats[this.chatActual] = c; else { this.chats.unshift(c); this.chatActual = 0; } this.guardarChatsEnLocalStorage(); } }
  
  generarTitulo(): string { const m = this.messages.find(m => m.role === 'user'); if (m) { const t = m.text; if (t && t.length > 30) return t.substring(0, 30) + '...'; if (t) return t; if (m.isImage) return '📷 Chat con imagen'; if (m.isAudio) return '🎤 Audio'; } return `Chat ${new Date().toLocaleString()}`; }
  
  cargarChat(i: number): void { if (this.messages.length > 1 && this.chatActual !== null) this.guardarChatActual(); this.stopSpeech(); const c = this.chats[i]; if (c) { this.messages = [...c.messages]; this.chatActual = i; this.historialCompleto = []; for (const m of this.messages) { if (m.role === 'user') this.historialCompleto.push({ role: 'user', content: m.text || '[Img/Audio]' }); else if (m.role === 'bot') { const p = (m.raw || m.text || '').replace(/<[^>]*>/g, ''); if (p.trim()) this.historialCompleto.push({ role: 'assistant', content: p }); } } this.showSidebar = false; } }
  
  eliminarChat(i: number, e: Event): void { e.stopPropagation(); if (confirm('¿Eliminar chat?')) { this.chats.splice(i, 1); this.guardarChatsEnLocalStorage(); if (this.chatActual === i) this.nuevoChat(); else if (this.chatActual !== null && this.chatActual > i) this.chatActual--; } }
  
  guardarChatsEnLocalStorage(): void { localStorage.setItem('chats_realitia', JSON.stringify(this.chats)); }
  
  cargarChatsDeLocalStorage(): void { const g = localStorage.getItem('chats_realitia'); if (g) try { this.chats = JSON.parse(g); } catch { this.chats = []; } }
}