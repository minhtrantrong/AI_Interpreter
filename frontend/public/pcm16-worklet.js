class PCM16Processor extends AudioWorkletProcessor {
  constructor(options) {
    super()
    this.targetSampleRate = options.processorOptions?.targetSampleRate ?? 16000
    this.frameSamples = options.processorOptions?.frameSamples ?? 1600
    this.ratio = sampleRate / this.targetSampleRate
    this.sourcePosition = 0
    this.tail = new Float32Array(0)
    this.output = new Int16Array(this.frameSamples)
    this.outputOffset = 0
  }

  process(inputs) {
    const channel = inputs[0]?.[0]
    if (!channel?.length) return true

    const combined = new Float32Array(this.tail.length + channel.length)
    combined.set(this.tail)
    combined.set(channel, this.tail.length)

    while (this.sourcePosition + 1 < combined.length) {
      const index = Math.floor(this.sourcePosition)
      const fraction = this.sourcePosition - index
      const sample = combined[index] + (combined[index + 1] - combined[index]) * fraction
      const clamped = Math.max(-1, Math.min(1, sample))
      this.output[this.outputOffset++] = clamped < 0 ? clamped * 32768 : clamped * 32767
      this.sourcePosition += this.ratio

      if (this.outputOffset === this.frameSamples) {
        const frame = this.output
        let energy = 0
        for (let i = 0; i < frame.length; i += 1) {
          const normalized = frame[i] / 32768
          energy += normalized * normalized
        }
        const level = Math.sqrt(energy / frame.length)
        this.port.postMessage({ type: 'audio', buffer: frame.buffer, level }, [frame.buffer])
        this.output = new Int16Array(this.frameSamples)
        this.outputOffset = 0
      }
    }

    const consumed = Math.floor(this.sourcePosition)
    this.tail = combined.slice(consumed)
    this.sourcePosition -= consumed
    return true
  }
}

registerProcessor('pcm16-processor', PCM16Processor)
