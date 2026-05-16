(() => {
  const messagesEl = document.getElementById('messages');
  const messagesInner = document.getElementById('messages-inner');
  const input = document.getElementById('message-input');
  const sendBtn = document.getElementById('send-btn');
  const stopBtn = document.getElementById('stop-btn');
  const sessionListEl = document.getElementById('session-list');
  const sessionNewBtn = document.getElementById('session-new-btn');
  const memoryListEl = document.getElementById('memory-list');
  const memoryClearBtn = document.getElementById('memory-clear-btn');
  const memoryPanel = document.getElementById('memory-panel');
  const memoryCollapseBtn = document.getElementById('memory-collapse-btn');
  const musicPanel = document.getElementById('music-panel');
  const musicCollapseBtn = document.getElementById('music-collapse-btn');
  const musicCollapsedNowPlayingEl = document.getElementById('music-collapsed-now-playing');
  const sidebar = document.getElementById('sidebar');
  const sidebarToggleBtn = document.getElementById('sidebar-toggle-btn');
  const ttsEnableBtn = document.getElementById('tts-enable-btn');
  const ttsStopBtn = document.getElementById('tts-stop-btn');

  window.appUi = { messagesEl, messagesInner, input, sendBtn };
  let currentSessionId = null;
  let creatingNewSession = false;
  let ttsAudio = null;
  let pendingVoicePlayback = null;
  let currentQueuePos = null;
  let _currentAbortController = null;

  // Phase 14: pending image attachment state
  let pendingImage = null; // { base64: string, mime: string, dataUrl: string } | null

  const imageUploadInput = document.getElementById('image-upload');
  const imageAttachBtn = document.getElementById('image-attach-btn');
  const imagePreviewStrip = document.getElementById('image-preview-strip');
  const imagePreviewThumb = document.getElementById('image-preview-thumb');
  const imageClearBtn = document.getElementById('image-clear-btn');
  const _MAX_IMAGE_BYTES = 25 * 1024 * 1024;

  function fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        // Strip the data-URI prefix — server expects raw base64
        const dataUrl = reader.result;
        const comma = dataUrl.indexOf(',');
        resolve({ base64: dataUrl.slice(comma + 1), dataUrl });
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  function clearPendingImage() {
    pendingImage = null;
    if (imagePreviewStrip) imagePreviewStrip.style.display = 'none';
    if (imagePreviewThumb) imagePreviewThumb.src = '';
    if (imageUploadInput) imageUploadInput.value = '';
  }

  if (imageAttachBtn && imageUploadInput) {
    imageAttachBtn.addEventListener('click', () => imageUploadInput.click());
    imageUploadInput.addEventListener('change', async () => {
      const file = imageUploadInput.files?.[0];
      if (!file) return;
      const allowed = ['image/png', 'image/jpeg', 'image/webp'];
      if (!allowed.includes(file.type)) {
        alert('Unsupported image type. Please use PNG, JPEG, or WebP.');
        imageUploadInput.value = '';
        return;
      }
      if (file.size > _MAX_IMAGE_BYTES) {
        alert(`Image too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Maximum is 25 MB.`);
        imageUploadInput.value = '';
        return;
      }
      const { base64, dataUrl } = await fileToBase64(file);
      pendingImage = { base64, mime: file.type, dataUrl };
      if (imagePreviewThumb) imagePreviewThumb.src = dataUrl;
      if (imagePreviewStrip) imagePreviewStrip.style.display = 'flex';
    });
  }
  if (imageClearBtn) {
    imageClearBtn.addEventListener('click', clearPendingImage);
  }

  // eslint-disable-next-line no-unused-vars
  function setTtsStatus(_text) { /* text removed; mic colour conveys state */ }

  function stopVoicePlayback() {
    pendingVoicePlayback = null;
    if (!ttsAudio) {
      if (ttsStopBtn) ttsStopBtn.style.display = 'none';
      setTtsStatus('voice idle');
      return;
    }
    ttsAudio.pause();
    ttsAudio.removeAttribute('src');
    ttsAudio.load();
    ttsAudio = null;
    if (ttsStopBtn) ttsStopBtn.style.display = 'none';
    setTtsStatus('voice idle');
  }

  async function playVoiceAudioFromText(text, endpoint = '/tts') {
    const trimmed = (text || '').trim();
    if (!trimmed) return;

    setTtsStatus('voice generating...');
    const resp = await (window.apiFetch || fetch)(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ text: trimmed }),
    });

    if (!resp.ok) {
      let details = `HTTP ${resp.status}`;
      try {
        const err = await resp.json();
        if (err?.error) details = err.error;
      } catch {
        // best effort
      }
      setTtsStatus('voice unavailable');
      throw new Error(`TTS failed: ${details}`);
    }

    const audioBytes = await resp.arrayBuffer();
    const blob = new Blob([audioBytes], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);

    stopVoicePlayback();
    ttsAudio = new Audio(url);
    ttsAudio.preload = 'auto';
    ttsAudio.autoplay = false;

    ttsAudio.addEventListener('ended', () => {
      if (ttsAudio) {
        URL.revokeObjectURL(ttsAudio.src);
      }
      ttsAudio = null;
      if (ttsStopBtn) ttsStopBtn.style.display = 'none';
      setTtsStatus('voice idle');
    }, { once: true });

    ttsAudio.addEventListener('error', () => {
      if (ttsAudio) {
        URL.revokeObjectURL(ttsAudio.src);
      }
      ttsAudio = null;
      if (ttsStopBtn) ttsStopBtn.style.display = 'none';
      setTtsStatus('voice error');
    }, { once: true });

    try {
      if (ttsStopBtn) ttsStopBtn.style.display = 'inline-block';
      setTtsStatus('voice speaking');
      // Try muted autoplay first (widest browser support), then unmute.
      ttsAudio.muted = true;
      await ttsAudio.play();
      ttsAudio.muted = false;
    } catch (err) {
      // Browser blocked autoplay; ask for a user gesture and keep payload queued.
      pendingVoicePlayback = { text: trimmed, endpoint };
      setTtsStatus('tap to enable voice');
      if (ttsEnableBtn) ttsEnableBtn.style.display = 'inline-block';
      if (ttsStopBtn) ttsStopBtn.style.display = 'none';
      if (ttsAudio) {
        URL.revokeObjectURL(ttsAudio.src);
      }
      ttsAudio = null;
      throw err;
    }
  }

  function isMobileLayout() {
    return window.matchMedia('(max-width: 900px)').matches;
  }

  function setPanelCollapsed(panelEl, collapseBtnEl, collapsed) {
    if (!panelEl || !collapseBtnEl) return;
    panelEl.classList.toggle('is-collapsed', collapsed);
    collapseBtnEl.textContent = collapsed ? '▸' : '▾';
    collapseBtnEl.setAttribute('aria-expanded', String(!collapsed));
    collapseBtnEl.title = `${collapsed ? 'Expand' : 'Collapse'} ${panelEl.id === 'music-panel' ? 'music' : 'memory'} section`;
  }

  function _bindCollapsiblePanels() {
    memoryCollapseBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      const collapsed = !memoryPanel?.classList.contains('is-collapsed');
      setPanelCollapsed(memoryPanel, memoryCollapseBtn, collapsed);
    });

    musicCollapseBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      const collapsed = !musicPanel?.classList.contains('is-collapsed');
      setPanelCollapsed(musicPanel, musicCollapseBtn, collapsed);
    });
  }

  function closeSidebar() {
    document.body.classList.remove('sidebar-open');
    sidebarToggleBtn?.setAttribute('aria-expanded', 'false');
  }

  function toggleSidebar() {
    if (isMobileLayout()) {
      const isOpen = document.body.classList.toggle('sidebar-open');
      sidebarToggleBtn?.setAttribute('aria-expanded', String(isOpen));
    } else {
      const isCollapsed = document.body.classList.toggle('sidebar-collapsed');
      sidebarToggleBtn?.setAttribute('aria-expanded', String(!isCollapsed));
    }
  }

  sidebarToggleBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    toggleSidebar();
  });

  document.addEventListener('click', (e) => {
    if (!isMobileLayout()) return;
    if (!document.body.classList.contains('sidebar-open')) return;
    const target = e.target;
    if (sidebar?.contains(target) || sidebarToggleBtn?.contains(target)) return;
    closeSidebar();
  });

  window.addEventListener('resize', () => {
    if (!isMobileLayout()) closeSidebar();
  });

  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 160) + 'px';
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  sendBtn.addEventListener('click', send);
  sessionNewBtn?.addEventListener('click', startNewChat);
  memoryClearBtn?.addEventListener('click', clearAllMemory);

  function setLocked(locked) {
    sendBtn.disabled = locked;
    input.disabled = locked;
    if (sessionNewBtn) sessionNewBtn.disabled = locked;
    if (memoryClearBtn) memoryClearBtn.disabled = locked;
    if (stopBtn) stopBtn.style.display = locked ? 'flex' : 'none';
    sendBtn.style.display = locked ? 'none' : 'flex';
  }

  stopBtn?.addEventListener('click', () => {
    _currentAbortController?.abort();
  });

  ttsEnableBtn?.addEventListener('click', async () => {
    const pending = pendingVoicePlayback;
    pendingVoicePlayback = null;
    ttsEnableBtn.style.display = 'none';
    if (!pending) {
      setTtsStatus('voice idle');
      return;
    }
    try {
      await playVoiceAudioFromText(pending.text, pending.endpoint);
    } catch {
      // keep state in status label
    }
  });

  ttsStopBtn?.addEventListener('click', () => {
    stopVoicePlayback();
  });

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function appendMessage(role, text = '', imageInfo = null) {
    document.getElementById('empty-state')?.remove();

    const wrapper = document.createElement('div');
    wrapper.className = `message ${role}`;

    // Phase 14: show image thumbnail in user message bubble
    if (role === 'user' && imageInfo) {
      const thumb = document.createElement('img');
      thumb.className = 'chat-image-thumb';
      thumb.src = imageInfo.dataUrl;
      thumb.alt = 'Attached image';
      wrapper.appendChild(thumb);
    }

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;

    wrapper.appendChild(bubble);
    messagesInner.appendChild(wrapper);
    scrollToBottom();
    return { wrapper, bubble };
  }

  function appendHistoryMessage(role, text = '') {
    const { bubble } = appendMessage(role, '');
    if (role === 'assistant') {
      bubble.innerHTML = marked.parse(text || '');
    } else {
      bubble.textContent = text || '';
    }
  }

  function renderEmptyState() {
    const empty = document.createElement('div');
    empty.id = 'empty-state';
    empty.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="44" height="44" viewBox="0 0 24 24"
           fill="none" stroke="currentColor" stroke-width="1.4"
           stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
      </svg>
      <span>Start a conversation</span>
    `;
    return empty;
  }

  function resetMessagesView() {
    messagesInner.innerHTML = '';
    messagesInner.appendChild(renderEmptyState());
    scrollToBottom();
  }

  function appendModelBadge(wrapper, modelName, intent, fallback) {
    const isCloud = modelName.toLowerCase().includes('claude');
    wrapper.querySelector('.model-badge')?.remove();

    const badge = document.createElement('span');
    badge.className = 'model-badge ' + (isCloud ? 'cloud' : 'local');

    const intentLabel = intent ? ` · ${intent}` : '';
    const fallbackLabel = fallback ? ' · fallback' : '';

    badge.textContent = modelName + intentLabel + fallbackLabel;
    badge.title = `Model: ${modelName}\nIntent: ${intent || 'unknown'}\nRoute: ${isCloud ? 'cloud' : 'local'}${fallback ? ' (fallback)' : ''}`;
    wrapper.appendChild(badge);
  }

  function appendMemoryBadge(wrapper, memory) {
    if (!memory) return;

    const status = memory.status || 'none';
    let label = '';
    if (status === 'saved' && memory.saved > 0) {
      label = `Memory: saved (${memory.saved})`;
    } else if (status === 'blocked-sensitive') {
      label = `Memory: blocked-sensitive (${memory.blocked})`;
    } else if (status === 'needs-confirmation') {
      label = `Memory: needs-confirmation (${memory.needs_confirmation})`;
    } else if (status === 'mixed-blocked-confirm') {
      label = `Memory: blocked ${memory.blocked}, needs-confirmation ${memory.needs_confirmation}`;
    } else if (status === 'do-not-remember') {
      label = 'Memory: not stored (as requested)';
    } else if (status === 'forgot') {
      label = `Memory: forgot (${memory.deleted || 0})`;
    } else if (status === 'no-target') {
      label = 'Memory: nothing to save yet';
    }

    if (!label && !memory.hint) return;

    wrapper.querySelector('.memory-badge')?.remove();
    const badge = document.createElement('span');
    badge.className = 'model-badge local memory-badge';
    badge.style.opacity = '0.75';
    badge.textContent = [label, memory.hint || ''].filter(Boolean).join(' ');
    wrapper.appendChild(badge);
  }

  function showVoiceError(msg) {
    document.getElementById('empty-state')?.remove();
    const wrapper = document.createElement('div');
    wrapper.className = 'message assistant';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.style.color = '#fa5252';
    bubble.textContent = `⚠ ${msg}`;

    wrapper.appendChild(bubble);
    messagesInner.appendChild(wrapper);
    scrollToBottom();
  }

  function renderSessions(sessions, activeId) {
    if (!sessionListEl) return;
    sessionListEl.innerHTML = '';
    if (!sessions.length) {
      const div = document.createElement('div');
      div.className = 'list-item';
      div.innerHTML = '<div class="list-item-title">No sessions yet</div>';
      sessionListEl.appendChild(div);
      return;
    }

    for (const session of sessions) {
      const item = document.createElement('div');
      item.className = 'list-item session-list-item' + (session.session_id === activeId ? ' active' : '');
      item.innerHTML = `
        <div class="session-row">
          <div class="list-item-title session-title">${_esc((session.preview || 'New session').slice(0, 80))}</div>
          <button class="memory-delete-btn session-delete-btn" data-sid="${session.session_id}" title="Delete session">&times;</button>
        </div>
      `;
      item.addEventListener('click', () => selectSession(session.session_id));
      item.querySelector('.session-delete-btn').addEventListener('click', async (e) => {
        e.stopPropagation();
        await deleteSession(session.session_id);
      });
      sessionListEl.appendChild(item);
    }
  }

  function renderMemory(items) {
    if (!memoryListEl) return;
    memoryListEl.innerHTML = '';
    if (!items.length) {
      const div = document.createElement('div');
      div.className = 'list-item';
      div.innerHTML = '<div class="list-item-title">No memory yet</div>';
      memoryListEl.appendChild(div);
      return;
    }

    for (const item of items) {
      const tier = String(item.tier || '').toLowerCase();
      const tierLabel = tier === 'episodic' ? 'Episodic' : tier === 'semantic' ? 'Semantic' : 'Working';
      const consolidatedLabel = tier === 'episodic'
        ? (item.consolidated ? 'Consolidated' : 'Pending consolidation')
        : '';

      const div = document.createElement('div');
      div.className = 'list-item';
      div.innerHTML = `
        <div class="list-item-title">${item.key}</div>
        <div class="list-item-meta">${(item.value || '').slice(0, 90)}</div>
        <div class="list-item-meta">${tierLabel}${consolidatedLabel ? ` · ${consolidatedLabel}` : ''}</div>
        <div class="memory-actions">
          <button class="memory-delete-btn" data-id="${item.id}">Delete</button>
        </div>
      `;
      div.querySelector('.memory-delete-btn')?.addEventListener('click', async (e) => {
        e.stopPropagation();
        await deleteMemory(item.id);
      });
      memoryListEl.appendChild(div);
    }
  }

  async function refreshSessions() {
    try {
      const resp = await (window.apiFetch || fetch)('/chat/sessions', { credentials: 'same-origin' });
      if (!resp.ok) return;
      const data = await resp.json();
      currentSessionId = data.current_session_id || currentSessionId;
      renderSessions(data.sessions || [], currentSessionId);
    } catch {
      // non-fatal
    }
  }

  async function refreshMemory() {
    try {
      const resp = await (window.apiFetch || fetch)('/memory?limit=200&offset=0', { credentials: 'same-origin' });
      if (!resp.ok) return;
      const data = await resp.json();
      renderMemory(data.items || []);
    } catch {
      // non-fatal
    }
  }

  // ── Music panel (Phase 8) ────────────────────────────────────────────────────

  function _esc(str) {
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  async function refreshNowPlaying() {
    const label = document.getElementById('now-playing-label');
    const btn = document.getElementById('music-play-pause-btn');
    const volumeInput = document.getElementById('music-volume');
    const volumeValue = document.getElementById('music-volume-value');
    if (!label) return;
    try {
      const resp = await (window.apiFetch || fetch)('/music/now_playing', { credentials: 'same-origin' });
      if (!resp.ok) return;
      const data = await resp.json();
      currentQueuePos = Number.isInteger(data.pos) ? data.pos : null;
      if (data.track && data.state !== 'stop') {
        const t = data.track;
        const parts = [t.artist, t.title].filter(Boolean);
        const nowPlayingText = parts.join(' — ');
        label.textContent = nowPlayingText;
        if (musicCollapsedNowPlayingEl) musicCollapsedNowPlayingEl.textContent = nowPlayingText;
        label.classList.remove('now-playing-idle');
        musicCollapsedNowPlayingEl?.classList.remove('now-playing-idle');
        if (btn) btn.textContent = data.state === 'play' ? '⏸' : '▶';
      } else {
        label.textContent = 'Nothing playing';
        if (musicCollapsedNowPlayingEl) musicCollapsedNowPlayingEl.textContent = 'Nothing playing';
        label.classList.add('now-playing-idle');
        musicCollapsedNowPlayingEl?.classList.add('now-playing-idle');
        if (btn) btn.textContent = '▶';
      }
      if (volumeInput && Number.isFinite(data.volume)) {
        const vol = Math.max(0, Math.min(100, Number(data.volume)));
        volumeInput.value = String(vol);
        if (volumeValue) volumeValue.textContent = `${vol}%`;
      }
    } catch {
      // non-fatal — MPD may not be running
    }
  }

  async function refreshQueue() {
    const list = document.getElementById('queue-list');
    if (!list) return;
    try {
      const resp = await (window.apiFetch || fetch)('/music/queue', { credentials: 'same-origin' });
      if (!resp.ok) return;
      const data = await resp.json();
      const items = data.queue || [];
      if (!items.length) {
        list.innerHTML = '<div class="list-item" style="color:var(--text-muted);font-style:italic;font-size:0.75rem">Queue empty</div>';
        return;
      }
      list.innerHTML = items.map(item => {
        const label = [item.artist, item.title].filter(Boolean).join(' — ') || 'Unknown track';
        const active = currentQueuePos === item.pos ? ' active-track' : '';
        return `<div class="list-item list-item-clickable${active}" data-pos="${item.pos}" title="Play this track"><span class="list-item-title">${_esc(label)}</span></div>`;
      }).join('');
      list.querySelectorAll('.list-item-clickable').forEach(el => {
        el.addEventListener('click', () => musicControl('play_pos', { pos: parseInt(el.dataset.pos, 10) }));
      });
    } catch {
      // non-fatal
    }
  }

  async function musicControl(action, extra = {}) {
    try {
      await (window.apiFetch || fetch)('/music/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ action, ...extra }),
      });
      // Brief delay so MPD state settles before polling.
      setTimeout(() => { refreshNowPlaying(); refreshQueue(); }, 400);
    } catch {
      // non-fatal
    }
  }

  // Wire music control buttons.
  (function _bindMusicControls() {
    const pp = document.getElementById('music-play-pause-btn');
    const next = document.getElementById('music-next-btn');
    const stop = document.getElementById('music-stop-btn');
    const volume = document.getElementById('music-volume');
    const volumeValue = document.getElementById('music-volume-value');
    if (pp) pp.addEventListener('click', async () => {
      // Toggle based on current label (▶ = resume, ⏸ = pause).
      const action = pp.textContent.trim() === '⏸' ? 'pause' : 'resume';
      await musicControl(action);
    });
    if (next) next.addEventListener('click', () => musicControl('next'));
    if (stop) stop.addEventListener('click', () => musicControl('stop'));
    if (volume) {
      volume.addEventListener('input', () => {
        if (volumeValue) volumeValue.textContent = `${volume.value}%`;
      });
      volume.addEventListener('change', () => {
        const value = Math.max(0, Math.min(100, parseInt(volume.value, 10) || 0));
        musicControl('set_volume', { volume: value });
      });
    }
  })();

  async function loadCurrentSessionMessages() {
    resetMessagesView();
    try {
      const resp = await (window.apiFetch || fetch)('/chat/session/messages', { credentials: 'same-origin' });
      if (!resp.ok) return;
      const data = await resp.json();
      currentSessionId = data.session_id || currentSessionId;
      const messages = data.messages || [];
      if (!messages.length) return;
      for (const msg of messages) {
        appendHistoryMessage(msg.role, msg.content || '');
      }
    } catch {
      // non-fatal
    }
  }

  async function selectSession(sessionId) {
    setLocked(true);
    closeSidebar();
    try {
      const resp = await (window.apiFetch || fetch)('/chat/session/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ session_id: sessionId }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      currentSessionId = sessionId;
      await loadCurrentSessionMessages();
      await refreshSessions();
    } catch (err) {
      appendMessage('assistant', `⚠ Unable to switch session: ${err.message}`);
    } finally {
      setLocked(false);
      input.focus();
    }
  }

  async function deleteMemory(id) {
    try {
      const resp = await (window.apiFetch || fetch)(`/memory/${encodeURIComponent(id)}`, {
        method: 'DELETE',
        credentials: 'same-origin',
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      await refreshMemory();
    } catch (err) {
      appendMessage('assistant', `⚠ Unable to delete memory: ${err.message}`);
    }
  }

  async function deleteSession(sessionId) {
    try {
      closeSidebar();
      const resp = await (window.apiFetch || fetch)(`/chat/sessions/${encodeURIComponent(sessionId)}`, {
        method: 'DELETE',
        credentials: 'same-origin',
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      if (sessionId === currentSessionId) {
        currentSessionId = data.active_session_id || null;
        await loadCurrentSessionMessages();
      }
      await refreshSessions();
    } catch (err) {
      appendMessage('assistant', `⚠ Unable to delete session: ${err.message}`);
    }
  }

  async function clearAllMemory() {
    if (!confirm('Clear all saved memory?')) return;
    closeSidebar();
    try {
      const resp = await (window.apiFetch || fetch)('/memory', {
        method: 'DELETE',
        credentials: 'same-origin',
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      await refreshMemory();
      appendMessage('assistant', 'Memory cleared.');
    } catch (err) {
      appendMessage('assistant', `⚠ Unable to clear memory: ${err.message}`);
    }
  }

  async function send(options = {}) {
    const source = options?.source === 'voice' ? 'voice' : 'text';
    const text = input.value.trim();
    if (!text && !pendingImage) return;
    closeSidebar();

    // Capture image before clearing
    const imageSnapshot = pendingImage ? { ...pendingImage } : null;
    appendMessage('user', text, imageSnapshot);
    input.value = '';
    input.style.height = 'auto';
    clearPendingImage();
    setLocked(true);

    const { wrapper, bubble } = appendMessage('assistant');
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    bubble.appendChild(cursor);

    let accumulated = '';
    let voiceMeta = null;
    const abortController = new AbortController();
    _currentAbortController = abortController;

    try {
      const resp = await (window.apiFetch || fetch)('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        signal: abortController.signal,
        body: JSON.stringify({
          message: text,
          source,
          ...(imageSnapshot && {
            image_base64: imageSnapshot.base64,
            image_mime: imageSnapshot.mime,
          }),
        }),
      });

      if (!resp.ok) {
        let errMsg = `Server error: HTTP ${resp.status}`;
        try {
          const errBody = await resp.clone().json();
          if (errBody?.error) errMsg = errBody.error;
        } catch { /* best effort */ }
        throw new Error(errMsg);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          const raw = line.slice(6).trim();
          if (raw === '[DONE]') continue;

          let parsed;
          try {
            parsed = JSON.parse(raw);
          } catch {
            continue;
          }

          if (parsed.model) {
            appendModelBadge(wrapper, parsed.model, parsed.intent, parsed.fallback);
          }

          if (parsed.notice) {
            const notice = document.createElement('span');
            notice.className = 'model-badge local';
            notice.style.opacity = '0.65';
            notice.style.marginRight = '0.4rem';
            notice.textContent = `⚠ ${parsed.notice}`;
            wrapper.appendChild(notice);
          }

          if (parsed.memory) {
            appendMemoryBadge(wrapper, parsed.memory);
          }

          if (parsed.voice) {
            voiceMeta = parsed.voice;
          }

          if (parsed.text) {
            accumulated += parsed.text;
            bubble.innerHTML = marked.parse(accumulated);
            bubble.appendChild(cursor);
            scrollToBottom();
          }
        }
      }

      if (source === 'voice' && voiceMeta?.tts_ready && voiceMeta?.tts_endpoint) {
        playVoiceAudioFromText(accumulated, voiceMeta.tts_endpoint).catch((err) => {
          console.warn('[tts] playback skipped:', err?.message || err);
        });
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        // user stopped generation — leave partial response as-is
      } else {
        bubble.textContent = `⚠ ${err.message}`;
      }
    } finally {
      _currentAbortController = null;
      cursor.remove();
      setLocked(false);
      input.focus();
      await refreshSessions();
      await refreshMemory();
      refreshNowPlaying();
      refreshQueue();
    }
  }
  async function startNewChat(event) {
      event?.preventDefault?.();
      event?.stopPropagation?.();
      if (creatingNewSession) return;

      // If the current session is already empty, don't create a duplicate.
      const isEmpty = messagesInner.querySelector('#empty-state') !== null &&
                      messagesInner.children.length === 1;
      if (isEmpty) {
        closeSidebar();
        input.focus();
        return;
      }

      creatingNewSession = true;
      if (sessionNewBtn) sessionNewBtn.disabled = true;
      setLocked(true);
      closeSidebar();
      try {
        const resp = await (window.apiFetch || fetch)('/chat/session/new', {
          method: 'POST',
          credentials: 'same-origin',
        });
        if (!resp.ok) throw new Error(`Server error: HTTP ${resp.status}`);
        const data = await resp.json();
        currentSessionId = data.session_id;
        resetMessagesView();
        await refreshSessions();
      } catch (err) {
        appendMessage('assistant', `⚠ Unable to create new chat session: ${err.message}`);
      } finally {
        creatingNewSession = false;
        setLocked(false);
        if (sessionNewBtn) sessionNewBtn.disabled = false;
        input.focus();
      }
  }

  async function bootstrap() {
    _bindCollapsiblePanels();
    await Promise.all([refreshSessions(), refreshMemory()]);
    await loadCurrentSessionMessages();
    refreshNowPlaying();
    refreshQueue();
    // Poll now-playing and queue every 10 s to keep the sidebar in sync.
    setInterval(() => { refreshNowPlaying(); refreshQueue(); }, 10_000);
  }

  void bootstrap();

  window.sendMessage = send;
  window.showVoiceError = showVoiceError;
  window.stopAssistantAudio = stopVoicePlayback;
})();
