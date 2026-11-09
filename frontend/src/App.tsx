import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import { useRealtimeTranslation } from "./useRealtimeTranslation";

const languages = [
  { value: "auto", label: "Tự động", short: "Auto" },
  { value: "vi", label: "Tiếng Việt", short: "VI" },
  { value: "en", label: "English", short: "EN" },
  { value: "ja", label: "日本語", short: "JA" },
  { value: "ko", label: "한국어", short: "KO" },
];

const asrModels = [
  { value: "small", label: "Whisper Small", note: "Nhanh" },
  { value: "turbo", label: "Large-v3 Turbo", note: "Chính xác hơn" },
];

const defaultAsrModel =
  import.meta.env.VITE_DEFAULT_ASR_MODEL === "turbo" ? "turbo" : "small";

const waveHeights = [
  24, 42, 18, 52, 32, 68, 45, 74, 38, 58, 82, 48, 66, 34, 72, 42, 54, 28, 48,
  20,
];
const TEXT_TRANSLATION_DEBOUNCE_MS = 600;

function ChevronDown() {
  return (
    <svg
      className="chevron-icon"
      width="10"
      height="6"
      viewBox="0 0 10 6"
      fill="none"
    >
      <path
        d="M1 1L5 5L9 1"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function AudioWaveform({
  audioLevel,
  isListening,
  className = "",
}: {
  audioLevel: number;
  isListening: boolean;
  className?: string;
}) {
  return (
    <div
      className={`waveform ${className} ${isListening ? "is-active" : ""}`.trim()}
      aria-label={
        isListening
          ? "Âm thanh đang được gửi tới backend"
          : "Microphone đang tạm dừng"
      }
    >
      {waveHeights.map((height, index) => {
        const liveHeight = Math.min(
          32,
          Math.max(4, height * (0.4 + audioLevel * 3)),
        );
        return (
          <span
            key={index}
            className="wave-bar"
            style={
              {
                "--wave-height": `${isListening ? liveHeight : height * 0.3}px`,
                "--wave-delay": `${index * -0.06}s`,
              } as React.CSSProperties
            }
          />
        );
      })}
    </div>
  );
}

function LanguageSelect({
  value,
  onChange,
  label,
  allowAuto = true,
}: {
  value: string;
  onChange: (value: string) => void;
  label: string;
  allowAuto?: boolean;
}) {
  const availableLanguages = allowAuto
    ? languages
    : languages.filter((language) => language.value !== "auto");
  const active =
    languages.find((language) => language.value === value) ?? languages[0];

  return (
    <label className="language-select">
      <span className="sr-only">{label}</span>
      <span className="language-badge">{active.short}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {availableLanguages.map((language) => (
          <option key={language.value} value={language.value}>
            {language.label}
          </option>
        ))}
      </select>
      <ChevronDown />
    </label>
  );
}

function formatDuration(seconds: number) {
  const minutes = Math.floor(seconds / 60)
    .toString()
    .padStart(2, "0");
  const remainder = (seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${remainder}`;
}

function languageLabel(code: string | null) {
  if (!code) return "Chưa nhận diện";
  return languages.find((language) => language.value === code)?.label ?? code;
}

function App() {
  const [sourceLanguage, setSourceLanguage] = useState("auto");
  const [targetLanguage, setTargetLanguage] = useState("en");
  const [asrModel, setAsrModel] = useState(defaultAsrModel);
  const [copied, setCopied] = useState(false);
  const [textInput, setTextInput] = useState("");
  const transcriptScrollRef = useRef<HTMLDivElement>(null);
  const textInputScrollRef = useRef<HTMLTextAreaElement>(null);
  const translationScrollRef = useRef<HTMLTextAreaElement>(null);

  const realtime = useRealtimeTranslation({
    sourceLanguage,
    targetLanguage,
    asrModel,
  });
  const submitTextRef = useRef(realtime.submitText);
  const resetRealtimeRef = useRef(realtime.reset);
  const lastSubmittedTextKeyRef = useRef("");
  const toggleListening = realtime.toggleListening;
  const isBusy =
    realtime.connectionState === "connecting" ||
    realtime.connectionState === "loading";

  // When listening: show speech transcript only
  // When not listening: show text input, or placeholder
  const isListening = realtime.isListening;
  const hasTranscript = !!realtime.transcript;
  const hasTextInput = !!textInput.trim();

  // Text for source panel
  const sourceText = isListening
    ? hasTranscript
      ? realtime.transcript
      : "Đang nghe..."
    : hasTextInput
      ? textInput
      : "";

  const sourcePlaceholder = isListening
    ? "Đang nghe..."
    : 'Nhấn "Bắt đầu nghe" và nói vào microphone, hoặc nhập text để dịch.';

  const displayedTranslation =
    realtime.translation || "Bản dịch realtime sẽ xuất hiện tại đây.";

  const wordRate = useMemo(() => {
    if (!realtime.transcript || realtime.elapsedSeconds === 0) return 0;
    const words = realtime.transcript.trim().split(/\s+/).length;
    return Math.round((words * 60) / realtime.elapsedSeconds);
  }, [realtime.elapsedSeconds, realtime.transcript]);

  const status = useMemo(() => {
    if (realtime.error)
      return {
        title: "Cần kiểm tra kết nối",
        detail: realtime.error,
        tone: "error",
      };
    if (realtime.connectionState === "loading")
      return {
        title: "Đang tải model local",
        detail:
          asrModel === "turbo"
            ? "Large-v3 Turbo trên GPU"
            : "Whisper Small trên GPU",
        tone: "loading",
      };
    if (realtime.connectionState === "connecting")
      return {
        title: "Đang kết nối backend",
        detail: "Socket đang sẵn sàng",
        tone: "loading",
      };
    if (realtime.isListening)
      return {
        title: "Đang phiên dịch",
        detail:
          realtime.latencyMs === null
            ? "Nhận âm thanh"
            : `Phản hồi ${realtime.latencyMs}ms`,
        tone: "ready",
      };
    if (realtime.connectionState === "ready")
      return {
        title: "Backend sẵn sàng",
        detail: "Nhấn để mở microphone",
        tone: "ready",
      };
    return {
      title: "Sẵn sàng bắt đầu",
      detail: "Backend local WebSocket",
      tone: "idle",
    };
  }, [
    asrModel,
    realtime.connectionState,
    realtime.error,
    realtime.isListening,
    realtime.latencyMs,
  ]);

  useEffect(() => {
    submitTextRef.current = realtime.submitText;
    resetRealtimeRef.current = realtime.reset;
  }, [realtime.reset, realtime.submitText]);

  useEffect(() => {
    if (isListening) return;

    const text = textInput.trim();
    if (!text) {
      if (lastSubmittedTextKeyRef.current) {
        lastSubmittedTextKeyRef.current = "";
        resetRealtimeRef.current();
      }
      return;
    }

    const requestKey = `${sourceLanguage}:${targetLanguage}:${text}`;
    const timer = window.setTimeout(() => {
      if (lastSubmittedTextKeyRef.current === requestKey) return;

      lastSubmittedTextKeyRef.current = requestKey;
      resetRealtimeRef.current();
      void submitTextRef.current(text);
    }, TEXT_TRANSLATION_DEBOUNCE_MS);

    return () => window.clearTimeout(timer);
  }, [isListening, sourceLanguage, targetLanguage, textInput]);

  useEffect(() => {
    if (!realtime.isListening) return;
    const frame = window.requestAnimationFrame(() => {
      transcriptScrollRef.current?.scrollTo({
        top: transcriptScrollRef.current.scrollHeight,
        behavior: "smooth",
      });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [realtime.transcript, realtime.isListening]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      translationScrollRef.current?.scrollTo({
        top: translationScrollRef.current.scrollHeight,
        behavior: "smooth",
      });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [realtime.translation]);

  useEffect(() => {
    const handleShortcut = (event: KeyboardEvent) => {
      if (
        event.code !== "Space" ||
        event.repeat ||
        event.target instanceof HTMLSelectElement ||
        event.target instanceof HTMLButtonElement ||
        event.target instanceof HTMLTextAreaElement
      )
        return;
      event.preventDefault();
      toggleListening();
    };

    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, [toggleListening]);

  const swapLanguages = () => {
    const nextSource = targetLanguage;
    const nextTarget = sourceLanguage === "auto" ? "vi" : sourceLanguage;
    setSourceLanguage(nextSource);
    setTargetLanguage(nextTarget);
  };

  const copyTranslation = async () => {
    if (!realtime.translation) return;
    await navigator.clipboard?.writeText(realtime.translation);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };

  const speakTranslation = () => {
    if (!realtime.translation || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(realtime.translation);
    utterance.lang = targetLanguage;
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-content">
          <div className="brand-group">
            <a className="brand" href="#top" aria-label="Lingo home">
              <span className="brand-icon" aria-hidden="true">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <rect
                    x="2"
                    y="6"
                    width="20"
                    height="12"
                    rx="2"
                    stroke="currentColor"
                    strokeWidth="2"
                  />
                  <path
                    d="M6 10V14M10 8V16M14 10V14M18 8V16"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                </svg>
              </span>
              <span className="brand-name">TalkSync</span>
            </a>
            <span className="header-slogan">
              <strong>Luôn luôn lắng nghe </strong> Lâu lâu không hiểu
            </span>
          </div>
          <div className="topbar-meta">
            <span className="privacy-badge">
              <span className="privacy-dot" />
              Xử lý 100% trên máy local
            </span>
          </div>
        </div>
      </header>

      <main id="top">
        <section className="translator" aria-label="Không gian dịch trực tiếp">
          <div className="panel source-panel">
            <div className="panel-header">
              <div className="panel-header-left">
                <span className="panel-label">Ngôn ngữ đầu vào</span>
                <LanguageSelect
                  value={sourceLanguage}
                  onChange={setSourceLanguage}
                  label="Chọn ngôn ngữ đầu vào"
                />
              </div>
              <span className="detected-badge">
                <span className="detected-dot" />
                {languageLabel(realtime.detectedLanguage)}
              </span>
            </div>

            <div className="panel-body">
              <div className="status-row">
                <span
                  className={`status-indicator ${isListening ? "listening" : ""}`}
                />
                <span className="status-text">
                  {isListening
                    ? "ĐANG LẮNG NGHE"
                    : isBusy
                      ? "KHỞI TẠO"
                      : "MIC TẠM DỪNG"}
                </span>
                <time className="elapsed-time">
                  {formatDuration(realtime.elapsedSeconds)}
                </time>
              </div>

              {/* Transcript / Text Input */}
              {isListening ? (
                <div
                  ref={transcriptScrollRef}
                  className={`transcript ${hasTranscript ? "" : "empty-copy"}`}
                  aria-live="polite"
                  tabIndex={0}
                >
                  {sourceText || sourcePlaceholder}
                </div>
              ) : (
                <textarea
                  ref={textInputScrollRef}
                  className="text-input-area"
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder="Nhập text, bản dịch sẽ tự động cập nhật..."
                  disabled={isBusy}
                  aria-label="Nhập text để dịch"
                />
              )}
            </div>

            <div className="panel-footer">
              <div className="footer-left">
                <span className="audio-info">
                  <span className="signal-bars">
                    <span />
                    <span />
                    <span />
                  </span>
                  {isListening ? "PCM16 · 16kHz" : "Micro sẵn sàng"}
                </span>
                <span className="word-rate">{wordRate} từ/phút</span>
              </div>
              <div className="footer-right">
                <label className="model-select">
                  <span className="model-label">STT</span>
                  <select
                    value={asrModel}
                    onChange={(event) => setAsrModel(event.target.value)}
                    disabled={realtime.isListening || isBusy}
                    aria-label="Chọn model nhận dạng giọng nói"
                  >
                    {asrModels.map((model) => (
                      <option key={model.value} value={model.value}>
                        {model.label} · {model.note}
                      </option>
                    ))}
                  </select>
                  <ChevronDown />
                </label>
              </div>
            </div>
          </div>

          <button
            className="swap-button"
            type="button"
            onClick={swapLanguages}
            aria-label="Đổi ngôn ngữ"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path
                d="M7 16L3 12L7 8M17 8L21 12L17 16M3 12H21"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>

          <div className="panel translation-panel">
            <div className="panel-header">
              <div className="panel-header-left">
                <span className="panel-label">Bản dịch</span>
                <LanguageSelect
                  value={targetLanguage}
                  onChange={setTargetLanguage}
                  label="Chọn ngôn ngữ bản dịch"
                  allowAuto={false}
                />
              </div>
              <span className="ai-badge">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                  <path
                    d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z"
                    fill="currentColor"
                  />
                </svg>
                NLLB-200 local
              </span>
            </div>

            <div className="panel-body">
              <textarea
                ref={translationScrollRef}
                className={`translation-text ${realtime.translation ? "" : "empty-copy"}`}
                readOnly
                aria-live="polite"
                tabIndex={0}
                value={displayedTranslation}
              />
              <span className="confidence-text">
                NLLB-200 INT8 · Không gửi dữ liệu lên cloud
              </span>
            </div>

            <div className="panel-footer translation-footer">
              <button
                className="clear-btn"
                type="button"
                onClick={() => {
                  setTextInput("");
                  realtime.reset();
                }}
              >
                Xóa nội dung
              </button>
              <div className="action-buttons">
                <button
                  type="button"
                  className="action-btn"
                  onClick={speakTranslation}
                  disabled={!realtime.translation}
                  aria-label="Phát bản dịch"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <path
                      d="M11 5L6 9H2V15H6L11 19V5Z"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <path
                      d="M15.54 8.46C16.4774 9.39764 17.0039 10.6692 17.0039 11.995C17.0039 13.3208 16.4774 14.5924 15.54 15.53M19.07 4.93C20.9447 6.80528 21.9979 9.34836 21.9979 12C21.9979 14.6516 20.9447 17.1947 19.07 19.07"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                    />
                  </svg>
                </button>
                <button
                  type="button"
                  className="action-btn copy-btn"
                  onClick={copyTranslation}
                  disabled={!realtime.translation}
                  aria-label="Sao chép bản dịch"
                >
                  {copied ? (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                      <path
                        d="M20 6L9 17L4 12"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  ) : (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                      <rect
                        x="9"
                        y="9"
                        width="13"
                        height="13"
                        rx="2"
                        stroke="currentColor"
                        strokeWidth="2"
                      />
                      <path
                        d="M5 15H4C2.89543 15 2 14.1046 2 13V4C2 2.89543 2.89543 2 4 2H13C14.1046 2 15 2.89543 15 4V5"
                        stroke="currentColor"
                        strokeWidth="2"
                      />
                    </svg>
                  )}
                  <span>{copied ? "Đã chép" : "Sao chép"}</span>
                </button>
              </div>
            </div>
          </div>
        </section>

        <section className="controls" aria-label="Điều khiển phiên dịch">
          <div className={`session-status status-${status.tone}`}>
            <span className="status-orb">
              <span className="orb-inner" />
            </span>
            <div className="status-info">
              <strong>{status.title}</strong>
              <span title={status.detail}>{status.detail}</span>
            </div>
          </div>

          <div className="record-control">
            <AudioWaveform
              audioLevel={realtime.audioLevel}
              isListening={isListening}
              className="waveform-main"
            />
            <button
              type="button"
              className={`record-btn ${isListening ? "recording" : ""}`}
              onClick={toggleListening}
              disabled={isBusy}
            >
              <svg
                className="mic-icon"
                width="18"
                height="24"
                viewBox="0 0 18 24"
                fill="none"
              >
                <rect
                  x="5"
                  y="2"
                  width="8"
                  height="14"
                  rx="4"
                  stroke="currentColor"
                  strokeWidth="2"
                />
                <path
                  d="M9 16V20M5 20H13"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
                <path
                  d="M3 9C3 12.866 5.686 16 9 16C12.314 16 15 12.866 15 9"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
              </svg>
              {isListening
                ? "Tạm dừng"
                : isBusy
                  ? "Đang tải..."
                  : "Bắt đầu nghe"}
            </button>
          </div>

          <p className="shortcut-hint">
            Nhấn <kbd>Space</kbd> để {isListening ? "tạm dừng" : "bắt đầu"}
          </p>
        </section>
      </main>
    </div>
  );
}

export default App;
