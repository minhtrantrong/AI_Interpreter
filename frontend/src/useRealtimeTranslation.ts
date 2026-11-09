import { useCallback, useEffect, useRef, useState } from 'react'

type ConnectionState = 'idle' | 'connecting' | 'loading' | 'ready' | 'error'

type ServerEvent = {
  type: string
  text?: string
  language?: string
  message?: string
  timestamp?: number
  asr_model?: string
}

type RealtimeOptions = {
  sourceLanguage: string
  targetLanguage: string
  asrModel: string
}

const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss' : 'ws'
const DEFAULT_WS_URL = import.meta.env.DEV
  ? `${WS_PROTOCOL}://${window.location.hostname}:8000/ws/realtime`
  : `${WS_PROTOCOL}://${window.location.host}/ws/realtime`
const WS_URL = import.meta.env.VITE_WS_URL || DEFAULT_WS_URL
const MAX_HISTORY_CHARS = 6000

function joinHistory(history: string, current: string) {
  const combined = [history.trim(), current.trim()].filter(Boolean).join('\n')
  if (combined.length <= MAX_HISTORY_CHARS) return combined
  return combined.slice(-MAX_HISTORY_CHARS).replace(/^\S*\s/, '')
}

export function useRealtimeTranslation({ sourceLanguage, targetLanguage, asrModel }: RealtimeOptions) {
  const [connectionState, setConnectionState] = useState<ConnectionState>('idle')
  const [isListening, setIsListening] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [translation, setTranslation] = useState('')
  const [detectedLanguage, setDetectedLanguage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [audioLevel, setAudioLevel] = useState(0)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const [activeAsrModel, setActiveAsrModel] = useState<string | null>(null)

  const socketRef = useRef<WebSocket | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const workletRef = useRef<AudioWorkletNode | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const gainRef = useRef<GainNode | null>(null)
  const listeningRef = useRef(false)
  const manuallyClosedRef = useRef(false)
  const committedTranscriptRef = useRef('')
  const committedTranslationRef = useRef('')
  const textInputBufferRef = useRef<string | null>(null)

  const handleServerEvent = useCallback((event: ServerEvent) => {
    if (event.timestamp) {
      setLatencyMs(Math.max(0, Math.round(Date.now() - event.timestamp * 1000)))
    }

    switch (event.type) {
      case 'session.loading':
      case 'session.model_loading':
        setConnectionState('loading')
        break
      case 'session.ready':
      case 'session.model':
        setConnectionState('ready')
        if (event.asr_model) setActiveAsrModel(event.asr_model)
        setError(null)
        // Check if there's buffered text input to submit
        if (textInputBufferRef.current) {
          const text = textInputBufferRef.current
          textInputBufferRef.current = null
          socketRef.current?.send(JSON.stringify({
            type: 'translate_text',
            text,
          }))
        }
        break
      case 'transcript.partial':
        if (event.text !== undefined) {
          setTranscript(joinHistory(committedTranscriptRef.current, event.text))
        }
        if (event.language) setDetectedLanguage(event.language)
        break
      case 'transcript.final':
        if (event.text) {
          committedTranscriptRef.current = joinHistory(
            committedTranscriptRef.current,
            event.text,
          )
          setTranscript(committedTranscriptRef.current)
        }
        if (event.language) setDetectedLanguage(event.language)
        break
      case 'translation.partial':
        if (event.text !== undefined) {
          setTranslation(joinHistory(committedTranslationRef.current, event.text))
        }
        break
      case 'translation.final':
        if (event.text) {
          committedTranslationRef.current = joinHistory(
            committedTranslationRef.current,
            event.text,
          )
          setTranslation(committedTranslationRef.current)
        }
        break
      case 'translation.error':
        setError(event.message || 'Không thể dịch câu vừa nhận.')
        break
      case 'error':
        setError(event.message || 'Backend không thể xử lý yêu cầu realtime.')
        setConnectionState('error')
        break
    }
  }, [])

  const connect = useCallback(() => {
    const current = socketRef.current
    if (current?.readyState === WebSocket.OPEN && connectionState === 'ready') {
      return Promise.resolve(current)
    }

    setConnectionState('connecting')
    setError(null)
    manuallyClosedRef.current = false

    return new Promise<WebSocket>((resolve, reject) => {
      const url = new URL(WS_URL)
      url.searchParams.set('source', sourceLanguage)
      url.searchParams.set('target', targetLanguage)

      const socket = new WebSocket(url)
      socketRef.current = socket

      socket.addEventListener('open', () => {
        setConnectionState('loading')
        socket.send(JSON.stringify({
          type: 'set_asr_model',
          model: asrModel,
        }))
        resolve(socket)
      })

      socket.addEventListener('message', (event) => {
        try {
          const data = JSON.parse(event.data as string)
          handleServerEvent(data)
        } catch {
          console.error('Failed to parse WebSocket message')
        }
      })

      socket.addEventListener('close', () => {
        setIsListening(false)
        if (!manuallyClosedRef.current) {
          setConnectionState('idle')
        }
      })

      socket.addEventListener('error', () => {
        setError('Không thể kết nối tới backend. Hãy chắc chắn backend đang chạy.')
        setConnectionState('error')
        reject(new Error('WebSocket connection failed'))
      })
    })
  }, [asrModel, handleServerEvent, sourceLanguage, targetLanguage, connectionState])

  const stopAudioCapture = useCallback(async () => {
    const stream = streamRef.current
    stream?.getTracks().forEach((track) => track.stop())
    streamRef.current = null

    const worklet = workletRef.current
    if (worklet) {
      worklet.disconnect()
      workletRef.current = null
    }

    const context = audioContextRef.current
    if (context) {
      await context.close()
      audioContextRef.current = null
    }

    sourceRef.current = null
    gainRef.current = null
  }, [])

  const toggleListening = useCallback(async () => {
    if (isListening) {
      listeningRef.current = false
      setIsListening(false)
      setElapsedSeconds(0)
      socketRef.current?.send(JSON.stringify({ type: 'commit' }))
      await stopAudioCapture()
      return
    }

    try {
      listeningRef.current = true

      const socket = await connect()

      if (!listeningRef.current) return

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      streamRef.current = stream

      const audioContext = new AudioContext({ sampleRate: 16000 })
      audioContextRef.current = audioContext

      await audioContext.audioWorklet.addModule('/pcm16-worklet.js')
      const worklet = new AudioWorkletNode(audioContext, 'pcm16-processor')
      workletRef.current = worklet

      const source = audioContext.createMediaStreamSource(stream)
      sourceRef.current = source

      const gain = audioContext.createGain()
      gain.gain.value = 1.0
      gainRef.current = gain

      source.connect(gain).connect(worklet).connect(audioContext.destination)

      worklet.port.onmessage = (event) => {
        if (!listeningRef.current) return
        const { type, buffer, level } = event.data as {
          type: string
          buffer: ArrayBuffer
          level: number
        }
        if (type !== 'audio' || !(buffer instanceof ArrayBuffer) || buffer.byteLength === 0) return

        setAudioLevel(Number.isFinite(level) ? level : 0)
        if (socket.readyState === WebSocket.OPEN) socket.send(buffer)
      }

      setIsListening(true)
    } catch (err) {
      listeningRef.current = false
      setIsListening(false)
      await stopAudioCapture()
      setError(err instanceof Error ? err.message : 'Không thể truy cập microphone.')
    }
  }, [connect, isListening, stopAudioCapture])

  const reset = useCallback(() => {
    committedTranscriptRef.current = ''
    committedTranslationRef.current = ''
    setTranscript('')
    setTranslation('')
    setDetectedLanguage(null)
    setElapsedSeconds(0)
    setAudioLevel(0)
    setLatencyMs(null)
    socketRef.current?.send(JSON.stringify({ type: 'reset', clear_context: false }))
  }, [])

  // Reconnect when language/model changes while listening
  useEffect(() => {
    if (socketRef.current?.readyState === WebSocket.OPEN && connectionState === 'ready') {
      socketRef.current.send(JSON.stringify({
        type: 'set_languages',
        source: sourceLanguage,
        target: targetLanguage,
      }))
    }
  }, [sourceLanguage, targetLanguage, connectionState])

  // Update model if changed
  useEffect(() => {
    if (
      socketRef.current?.readyState === WebSocket.OPEN
      && connectionState === 'ready'
      && activeAsrModel !== asrModel
    ) {
      setConnectionState('loading')
      socketRef.current.send(JSON.stringify({
        type: 'set_asr_model',
        model: asrModel,
      }))
    }
  }, [activeAsrModel, asrModel, connectionState])

  useEffect(() => {
    if (!isListening) return
    const timer = window.setInterval(() => setElapsedSeconds((value) => value + 1), 1000)
    return () => window.clearInterval(timer)
  }, [isListening])

  useEffect(() => {
    if (connectionState === 'error' && listeningRef.current) {
      void stopAudioCapture()
    }
  }, [connectionState, stopAudioCapture])

  useEffect(() => () => {
    manuallyClosedRef.current = true
    listeningRef.current = false
    streamRef.current?.getTracks().forEach((track) => track.stop())
    void audioContextRef.current?.close()
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: 'end' }))
      socketRef.current.close(1000)
    }
  }, [])

  // Function to submit text for translation
  const submitText = useCallback(async (text: string) => {
    if (!text.trim()) return

    try {
      const socket = await connect()
      // Buffer the text until connected
      if (socket.readyState !== WebSocket.OPEN) {
        textInputBufferRef.current = text.trim()
        return
      }
      socket.send(JSON.stringify({
        type: 'translate_text',
        text: text.trim(),
      }))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Không thể gửi text tới backend.')
    }
  }, [connect])

  return {
    connectionState,
    isListening,
    transcript,
    translation,
    detectedLanguage,
    error,
    audioLevel,
    elapsedSeconds,
    latencyMs,
    activeAsrModel,
    toggleListening,
    reset,
    submitText,
  }
}
