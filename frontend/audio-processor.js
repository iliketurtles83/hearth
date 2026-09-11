/**
 * AudioWorkletProcessor: audio-processor
 *
 * Browsers (Chrome, Firefox, ...) ignore the `sampleRate` option on
 * AudioContext and run the graph at the device's hardware rate (typically
 * 44.1/48 kHz). openWakeWord — and the 16 kHz WAV sent to /transcribe — both
 * assume 16 kHz. So we resample the incoming audio to 16 kHz here (linear
 * interpolation, driven by the worklet's real `sampleRate`) instead of
 * assuming the context is already 16 kHz.
 *
 * Resampled samples are accumulated into a 1 280-sample frame (80 ms @
 * 16 kHz) and posted to the main thread as a transferable Int16Array, which
 * is the format expected by openWakeWord.
 */

const TARGET_RATE = 16000;
const FRAME_SIZE = 1280; // 80 ms @ 16 kHz

class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // > 1 when the context runs faster than 16 kHz (the common case).
    this._ratio = sampleRate / TARGET_RATE;
    this._input = []; // carry-over input samples not yet fully consumed
    this._pos = 0; // fractional read position within this._input
    this._out = new Float32Array(FRAME_SIZE);
    this._outIdx = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel || channel.length === 0) return true;

    for (let i = 0; i < channel.length; i++) this._input.push(channel[i]);

    // Produce 16 kHz output samples by linear interpolation across the
    // carry-over buffer, advancing `ratio` input samples per output sample.
    const need = FRAME_SIZE - this._outIdx;
    let idx = this._pos;
    let produced = 0;
    while (produced < need && idx + 1 < this._input.length) {
      const i0 = idx | 0;
      const frac = idx - i0;
      this._out[this._outIdx + produced] =
        this._input[i0] * (1 - frac) + this._input[i0 + 1] * frac;
      idx += this._ratio;
      produced++;
    }
    this._outIdx += produced;

    // Advance the read position and drop the fully-consumed samples.
    const consumed = idx | 0;
    this._pos = idx - consumed;
    if (consumed > 0) this._input = this._input.slice(consumed);

    if (this._outIdx >= FRAME_SIZE) {
      // Convert float32 [-1, 1] → int16 and transfer to main thread
      const int16 = new Int16Array(FRAME_SIZE);
      for (let j = 0; j < FRAME_SIZE; j++) {
        const s = Math.max(-1, Math.min(1, this._out[j]));
        int16[j] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      this.port.postMessage(int16.buffer, [int16.buffer]);
      this._out = new Float32Array(FRAME_SIZE);
      this._outIdx = 0;
    }
    return true;
  }
}

registerProcessor('audio-processor', AudioProcessor);
