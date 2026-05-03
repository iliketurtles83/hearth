(() => {
  const micBtn = document.getElementById('mic-btn');
  const voiceStatus = document.getElementById('voice-status');
  const input = window.appUi?.input || document.getElementById('message-input');

  let voiceState = 'off';

  function setVoiceState(state) {
    voiceState = state;
    if (voiceStatus) {
      voiceStatus.className = state === 'off' ? '' : state === 'sleeping' ? 'sleeping' : state === 'recording' ? 'recording' : 'transcribing';
    }
    micBtn.classList.toggle('active', state !== 'off');
  }

  function encodeWAV(samples) {
    const sampleRate = 16000;
    const numChannels = 1;
    const bitDepth = 16;
    const byteRate = sampleRate * numChannels * (bitDepth / 8);
    const blockAlign = numChannels * (bitDepth / 8);
    const dataLen = samples.length * 2;
    const buf = new ArrayBuffer(44 + dataLen);
    const view = new DataView(buf);

    const writeStr = (offset, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };

    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + dataLen, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeStr(36, 'data');
    view.setUint32(40, dataLen, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }

    return new Blob([buf], { type: 'audio/wav' });
  }

  async function transcribeAndSend(float32Audio) {
    setVoiceState('transcribing');
    try {
      const wav = encodeWAV(float32Audio);
      const form = new FormData();
      form.append('audio', wav, 'utterance.wav');

      const resp = await (window.apiFetch || fetch)('/transcribe', { method: 'POST', body: form });
      if (!resp.ok) throw new Error(`Transcription failed: HTTP ${resp.status}`);

      const { text } = await resp.json();
      if (text) {
        input.value = text;
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 160) + 'px';
        await window.sendMessage({ source: 'voice' });
      }
    } catch (err) {
      console.error('[voice] transcribe error:', err);
    } finally {
      setVoiceState('sleeping');
    }
  }

  const CAPTURE_MAX_MS = 15000;
  const RMS_SPEECH_THRESHOLD = 0.015;
  const MIN_SPEECH_FRAMES = 3;
  const END_SILENCE_FRAMES = 8;

  let _captureActive = false;
  let _capturedChunks = [];
  let _speechFrames = 0;
  let _silenceFrames = 0;
  let _captureTimeoutId = null;

  function int16ToFloat32(samples) {
    const out = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) out[i] = samples[i] / 32768;
    return out;
  }

  function combineFloat32Chunks(chunks) {
    const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const combined = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }
    return combined;
  }

  async function finishUtteranceCapture() {
    if (!_captureActive) return;

    _captureActive = false;
    if (_captureTimeoutId) {
      clearTimeout(_captureTimeoutId);
      _captureTimeoutId = null;
    }

    const enoughSpeech = _speechFrames >= MIN_SPEECH_FRAMES;
    const audio = enoughSpeech ? combineFloat32Chunks(_capturedChunks) : null;

    _capturedChunks = [];
    _speechFrames = 0;
    _silenceFrames = 0;

    if (audio && audio.length > 0) {
      await transcribeAndSend(audio);
    } else {
      setVoiceState('sleeping');
    }
  }

  function beginUtteranceCapture() {
    _captureActive = true;
    _capturedChunks = [];
    _speechFrames = 0;
    _silenceFrames = 0;
    setVoiceState('recording');

    _captureTimeoutId = setTimeout(() => {
      finishUtteranceCapture().catch((err) => {
        console.error('[voice] capture finalize error:', err);
        window.showVoiceError?.(`Voice capture failed: ${err.message}`);
        setVoiceState('sleeping');
      });
    }, CAPTURE_MAX_MS);
  }

  function processUtteranceFrame(int16Frame) {
    if (!_captureActive) return;

    const floatFrame = int16ToFloat32(int16Frame);
    _capturedChunks.push(floatFrame);

    let energy = 0;
    for (let i = 0; i < floatFrame.length; i++) energy += floatFrame[i] * floatFrame[i];
    const rms = Math.sqrt(energy / floatFrame.length);

    if (rms >= RMS_SPEECH_THRESHOLD) {
      _speechFrames += 1;
      _silenceFrames = 0;
    } else if (_speechFrames > 0) {
      _silenceFrames += 1;
    }

    if (_speechFrames >= MIN_SPEECH_FRAMES && _silenceFrames >= END_SILENCE_FRAMES) {
      finishUtteranceCapture().catch((err) => {
        console.error('[voice] capture finalize error:', err);
        window.showVoiceError?.(`Voice capture failed: ${err.message}`);
        setVoiceState('sleeping');
      });
    }
  }

  let _audioCtx = null;
  let _workletNode = null;
  let _ws = null;
  let _micStream = null;
  let _wakeGuardUntil = 0;
  let _reconnectDelay = 1000;
  const _MAX_RECONNECT = 30000;
  let _reconnectTimerId = null;
  let _manualStop = false;

  function interruptAssistantAudio() {
    const stopAudio = window.stopAssistantAudio;
    if (typeof stopAudio !== 'function') return 0;

    const t0 = performance.now();
    try {
      stopAudio();
    } catch (err) {
      console.warn('[voice] failed to interrupt assistant audio:', err);
    }
    return performance.now() - t0;
  }

  function connectWakeWebSocket() {
    const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
    _ws = new WebSocket(`${wsProto}://${location.host}/ws/wake`);
    _ws.binaryType = 'arraybuffer';

    _ws.onopen = () => {
      _reconnectDelay = 1000;
    };

    _ws.onmessage = async (evt) => {
      const msg = JSON.parse(evt.data);
      if (msg.event === 'wake' && voiceState === 'sleeping' && Date.now() > _wakeGuardUntil) {
        const interruptMs = interruptAssistantAudio();
        console.log('[voice] wake word detected, score:', msg.score);
        if (interruptMs > 0) {
          console.log(`[voice] barge-in interrupted assistant audio in ${interruptMs.toFixed(1)}ms`);
        }
        _wakeGuardUntil = Date.now() + 1500;
        beginUtteranceCapture();
      }
    };

    _ws.onclose = (evt) => {
      console.warn('[voice] WebSocket closed - code:', evt.code, 'reason:', evt.reason || '(none)');
      _ws = null;

      if (_manualStop || voiceState === 'off') return;

      const delay = _reconnectDelay;
      _reconnectDelay = Math.min(_reconnectDelay * 2, _MAX_RECONNECT);
      console.log(`[voice] reconnecting websocket in ${delay}ms...`);

      _reconnectTimerId = setTimeout(() => {
        if (_manualStop || voiceState === 'off' || !_audioCtx || !_workletNode) return;
        connectWakeWebSocket();
      }, delay);
    };
  }

  async function startVoice() {
    if (location.protocol === 'file:') {
      alert('Open this page via HTTPS (e.g. https://<host>) — voice features require a secure connection.');
      return;
    }

    if (!window.isSecureContext || !navigator.mediaDevices) {
      window.showVoiceError?.('Voice input requires a secure connection. Access this assistant via HTTPS, or from localhost.');
      return;
    }

    try {
      _manualStop = false;
      if (_reconnectTimerId) {
        clearTimeout(_reconnectTimerId);
        _reconnectTimerId = null;
      }

      _micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
        video: false,
      });

      _audioCtx = new AudioContext({ sampleRate: 16000 });
      await _audioCtx.audioWorklet.addModule(new URL('/static/audio-processor.js', window.location.href));

      const source = _audioCtx.createMediaStreamSource(_micStream);
      _workletNode = new AudioWorkletNode(_audioCtx, 'audio-processor');
      source.connect(_workletNode);

      connectWakeWebSocket();

      _workletNode.port.onmessage = (evt) => {
        const frame = new Int16Array(evt.data);

        if (_captureActive) processUtteranceFrame(frame);

        if (_ws && _ws.readyState === WebSocket.OPEN && voiceState === 'sleeping' && Date.now() > _wakeGuardUntil) {
          _ws.send(evt.data);
        }
      };

      setVoiceState('sleeping');
    } catch (err) {
      console.error('[voice] failed to start:', err);
      stopVoice();
      const isPermission = err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError';
      window.showVoiceError?.(isPermission ? 'Microphone access was denied. Please allow microphone access and try again.' : `Could not start voice input: ${err.message}`);
    }
  }

  function stopVoice() {
    _manualStop = true;

    if (_reconnectTimerId) {
      clearTimeout(_reconnectTimerId);
      _reconnectTimerId = null;
    }

    _ws?.close();
    _ws = null;
    _workletNode?.disconnect();
    _workletNode = null;
    _audioCtx?.close();
    _audioCtx = null;
    _micStream?.getTracks().forEach((t) => t.stop());
    _micStream = null;

    _captureActive = false;
    _capturedChunks = [];
    _speechFrames = 0;
    _silenceFrames = 0;

    if (_captureTimeoutId) {
      clearTimeout(_captureTimeoutId);
      _captureTimeoutId = null;
    }

    setVoiceState('off');
  }

  micBtn.addEventListener('click', async () => {
    if (voiceState === 'off') {
      await startVoice();
    } else {
      stopVoice();
    }
  });
})();
