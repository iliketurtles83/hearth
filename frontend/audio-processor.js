/**
 * AudioWorkletProcessor: audio-processor
 *
 * Receives 128-sample Float32 chunks from the Web Audio graph (at whatever
 * sample rate the AudioContext was created with — we use 16 000 Hz so no
 * resampling is needed here).
 *
 * Accumulates chunks into a 1 280-sample ring buffer (80 ms @ 16 kHz) and
 * posts each full frame to the main thread as a transferable Int16Array,
 * which is the format expected by openWakeWord.
 */

const FRAME_SIZE = 1280; // 80 ms @ 16 kHz

class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Float32Array(FRAME_SIZE);
    this._offset = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    let i = 0;
    while (i < channel.length) {
      const remaining = FRAME_SIZE - this._offset;
      const toCopy = Math.min(remaining, channel.length - i);
      this._buffer.set(channel.subarray(i, i + toCopy), this._offset);
      this._offset += toCopy;
      i += toCopy;

      if (this._offset === FRAME_SIZE) {
        // Convert float32 [-1, 1] → int16 and transfer to main thread
        const int16 = new Int16Array(FRAME_SIZE);
        for (let j = 0; j < FRAME_SIZE; j++) {
          const s = Math.max(-1, Math.min(1, this._buffer[j]));
          int16[j] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        this.port.postMessage(int16.buffer, [int16.buffer]);
        this._buffer = new Float32Array(FRAME_SIZE);
        this._offset = 0;
      }
    }
    return true;
  }
}

registerProcessor('audio-processor', AudioProcessor);
