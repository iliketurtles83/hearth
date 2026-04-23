const micBtn      = document.getElementById('mic-btn');
    const voiceStatus = document.getElementById('voice-status');

    // ── State machine ───────────────────────────────────────────────────────
    // States: 'off' | 'sleeping' | 'recording' | 'transcribing'
    let voiceState = 'off';

    const STATUS_LABELS = {
      off:          '',
      sleeping:     'listening for "Computer,"',
      recording:    'recording…',
      transcribing: 'transcribing…',
    };

    function setVoiceState(state) {
      voiceState = state;
      voiceStatus.className  = state === 'off' ? '' : state === 'sleeping' ? 'sleeping' :
                               state === 'recording' ? 'recording' : 'transcribing';
      voiceStatus.textContent = STATUS_LABELS[state] ?? '';
      micBtn.classList.toggle('active', state !== 'off');
    }

    // ── WAV encoder ─────────────────────────────────────────────────────────
    // Converts a Float32Array @16kHz into a WAV Blob.
    function encodeWAV(samples) {
      const sampleRate  = 16000;
      const numChannels = 1;
      const bitDepth    = 16;
      const byteRate    = sampleRate * numChannels * (bitDepth / 8);
      const blockAlign  = numChannels * (bitDepth / 8);
      const dataLen     = samples.length * 2;
      const buf         = new ArrayBuffer(44 + dataLen);
      const view        = new DataView(buf);

      const writeStr = (offset, str) => { for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i)); };
      writeStr(0,  'RIFF');                              // ChunkID
      view.setUint32(4,  36 + dataLen, true);           // ChunkSize
      writeStr(8,  'WAVE');                              // Format
      writeStr(12, 'fmt ');                              // Subchunk1ID
      view.setUint32(16, 16, true);                     // Subchunk1Size
      view.setUint16(20, 1,  true);                     // AudioFormat (PCM)
      view.setUint16(22, numChannels, true);            // NumChannels
      view.setUint32(24, sampleRate,  true);            // SampleRate
      view.setUint32(28, byteRate,    true);            // ByteRate
      view.setUint16(32, blockAlign,  true);            // BlockAlign
      view.setUint16(34, bitDepth,    true);            // BitsPerSample
      writeStr(36, 'data');                             // Subchunk2ID
      view.setUint32(40, dataLen, true);                // Subchunk2Size

      // PCM samples
      let offset = 44;
      for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        offset += 2;
      }
      return new Blob([buf], { type: 'audio/wav' });
    }

    // ── Transcription ───────────────────────────────────────────────────────
    async function transcribeAndSend(float32Audio) {
      setVoiceState('transcribing');
      try {
        const wav  = encodeWAV(float32Audio);
        const form = new FormData();
        form.append('audio', wav, 'utterance.wav');
        const resp = await fetch('/transcribe', { method: 'POST', body: form });
        if (!resp.ok) throw new Error(`Transcription failed: HTTP ${resp.status}`);
        const { text } = await resp.json();
        if (text) {
          input.value = text;
          input.style.height = 'auto';
          input.style.height = Math.min(input.scrollHeight, 160) + 'px';
          send();
        }
      } catch (err) {
        console.error('[voice] transcribe error:', err);
      } finally {
        setVoiceState('sleeping'); // return to wake-word listening
      }
    }

    // ── Utterance capture from existing worklet frames ─────────────────────
    const CAPTURE_MAX_MS = 15000;
    const RMS_SPEECH_THRESHOLD = 0.015;
    const MIN_SPEECH_FRAMES = 3;        // ~240ms speech before we accept audio
    const END_SILENCE_FRAMES = 8;       // ~640ms silence to stop capture
    let _captureActive = false;
    let _capturedChunks = [];
    let _speechFrames = 0;
    let _silenceFrames = 0;
    let _captureTimeoutId = null;

    function int16ToFloat32(samples) {
      const out = new Float32Array(samples.length);
      for (let i = 0; i < samples.length; i++) {
        out[i] = samples[i] / 32768;
      }
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
        finishUtteranceCapture().catch(err => {
          console.error('[voice] capture finalize error:', err);
          showVoiceError(`Voice capture failed: ${err.message}`);
          setVoiceState('sleeping');
        });
      }, CAPTURE_MAX_MS);
    }

    function processUtteranceFrame(int16Frame) {
      if (!_captureActive) return;

      const floatFrame = int16ToFloat32(int16Frame);
      _capturedChunks.push(floatFrame);

      let energy = 0;
      for (let i = 0; i < floatFrame.length; i++) {
        energy += floatFrame[i] * floatFrame[i];
      }
      const rms = Math.sqrt(energy / floatFrame.length);

      if (rms >= RMS_SPEECH_THRESHOLD) {
        _speechFrames += 1;
        _silenceFrames = 0;
      } else if (_speechFrames > 0) {
        _silenceFrames += 1;
      }

      if (_speechFrames >= MIN_SPEECH_FRAMES && _silenceFrames >= END_SILENCE_FRAMES) {
        finishUtteranceCapture().catch(err => {
          console.error('[voice] capture finalize error:', err);
          showVoiceError(`Voice capture failed: ${err.message}`);
          setVoiceState('sleeping');
        });
      }
    }

    // ── Wake-word WebSocket ─────────────────────────────────────────────────
    let _audioCtx   = null;
    let _workletNode = null;
    let _ws         = null;
    let _micStream  = null;
    let _wakeGuardUntil  = 0;        // timestamp: ignore wake + frame send until this time
    let _reconnectDelay  = 1000;     // ms; doubled on each failed reconnect
    const _MAX_RECONNECT = 30000;    // backoff cap (30 s)

    async function startVoice() {
      if (location.protocol === 'file:') {
        alert('Open this page over HTTP (e.g. http://<host>:8000) — voice features require a server.');
        return;
      }

      // navigator.mediaDevices is undefined on HTTP in mobile browsers (requires HTTPS)
      if (!window.isSecureContext || !navigator.mediaDevices) {
        showVoiceError(
          'Voice input requires a secure connection. ' +
          'Access this assistant via HTTPS, or from localhost.'
        );
        return;
      }

      try {
        // 1. Mic stream — explicit constraints improve reliability on Linux
        _micStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount:     1,
            sampleRate:       16000,
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl:  false,
          },
          video: false,
        });

        // 2. AudioContext @16kHz (all modern browsers support this)
        _audioCtx = new AudioContext({ sampleRate: 16000 });
        await _audioCtx.audioWorklet.addModule(`${location.origin}/audio-processor.js`);

        const source = _audioCtx.createMediaStreamSource(_micStream);
        _workletNode = new AudioWorkletNode(_audioCtx, 'audio-processor');
        source.connect(_workletNode);
        // Don't connect worklet to destination — we don't want to hear ourselves

        // 3. Open WebSocket to wake-word server
        const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
        _ws = new WebSocket(`${wsProto}://${location.host}/ws/wake`);
        _ws.binaryType = 'arraybuffer';

        _ws.onopen = () => {
          _reconnectDelay = 1000; // reset backoff on successful connection
        };

        _ws.onmessage = async (evt) => {
          const msg = JSON.parse(evt.data);
          if (msg.event === 'wake' && voiceState === 'sleeping' && Date.now() > _wakeGuardUntil) {
            console.log('[voice] wake word detected, score:', msg.score);
            _wakeGuardUntil = Date.now() + 1500; // 1.5 s post-wake guard window
            beginUtteranceCapture();
          }
        };

        _ws.onclose = (evt) => {
          console.warn('[voice] WebSocket closed — code:', evt.code, 'reason:', evt.reason || '(none)');
          if (voiceState !== 'off') {
            const delay = _reconnectDelay;
            _reconnectDelay = Math.min(_reconnectDelay * 2, _MAX_RECONNECT);
            console.log(`[voice] reconnecting in ${delay}ms…`);
            setTimeout(startVoice, delay);
          }
        };

        // 4. Pipe AudioWorklet frames → WebSocket
        _workletNode.port.onmessage = (evt) => {
          const frame = new Int16Array(evt.data);

          if (_captureActive) {
            processUtteranceFrame(frame);
          }

          if (_ws.readyState === WebSocket.OPEN && voiceState === 'sleeping' && Date.now() > _wakeGuardUntil) {
            _ws.send(evt.data); // transferable ArrayBuffer (Int16)
          }
        };

        setVoiceState('sleeping');
      } catch (err) {
        console.error('[voice] failed to start:', err);
        stopVoice();
        const isPermission = err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError';
        showVoiceError(isPermission
          ? 'Microphone access was denied. Please allow microphone access and try again.'
          : `Could not start voice input: ${err.message}`);
      }
    }

    function stopVoice() {
      _ws?.close();
      _ws = null;
      _workletNode?.disconnect();
      _workletNode = null;
      _audioCtx?.close();
      _audioCtx = null;
      _micStream?.getTracks().forEach(t => t.stop());
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

    // ── Mic button toggle ───────────────────────────────────────────────────
    micBtn.addEventListener('click', async () => {
      if (voiceState === 'off') {
        await startVoice();
      } else {
        stopVoice();
      }
    });