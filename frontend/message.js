(() => {
  const messagesEl = document.getElementById('messages');
  const messagesInner = document.getElementById('messages-inner');
  const input = document.getElementById('message-input');
  const sendBtn = document.getElementById('send-btn');
  const stopBtn = document.getElementById('stop-btn');
  const sessionListEl = document.getElementById('session-list');
  const sessionNewBtn = document.getElementById('session-new-btn');
  const memoryModal = document.getElementById('memory-modal');
  const memoryModalList = document.getElementById('memory-modal-list');
  const memoryModalCount = document.getElementById('memory-modal-count');
  const memoryModalClearBtn = document.getElementById('memory-modal-clear-btn');
  const memoryModalConsolidateBtn = document.getElementById('memory-modal-consolidate-btn');
  const memoryModalCloseBtn = document.getElementById('memory-modal-close-btn');
  const sessionsPanel = document.getElementById('sessions-panel');
  const sessionsCollapseBtn = document.getElementById('sessions-collapse-btn');
  const musicPanel = document.getElementById('music-panel');
  const musicCollapseBtn = document.getElementById('music-collapse-btn');
  const musicCollapsedNowPlayingEl = document.getElementById('music-collapsed-now-playing');
  const sidebar = document.getElementById('sidebar');
  const sidebarToggleBtn = document.getElementById('sidebar-toggle-btn');
  const ttsEnableBtn = document.getElementById('tts-enable-btn');
  const ttsStopBtn = document.getElementById('tts-stop-btn');
  const authUsernameBtn = document.getElementById('auth-username');

  window.appUi = { messagesEl, messagesInner, input, sendBtn };
  let currentSessionId = null;
  let creatingNewSession = false;
  let ttsAudio = null;
  let pendingVoicePlayback = null;
  let currentQueuePos = null;
  let _currentAbortController = null;
  const _REASONING_PREF_KEY = 'ui.showReasoning';
  const _THEME_PREF_KEY = 'ui.theme';
  let _showReasoning = true;
  let _theme = 'dark';
  let _sessionMenuEl = null;
  let _sessionMenuSid = null;
  let _sessionMenuTitle = '';
  let _settingsMenuEl = null;
  let _themeMenuItem = null;
  let _reasoningMenuItem = null;
  let _beetsMenuItem = null;
  let _consolidateMenuItem = null;
  let _settingsActionBusy = false;

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

  function _setReasoningVisible(visible) {
    _showReasoning = !!visible;
    document.body.classList.toggle('hide-reasoning', !_showReasoning);
    if (_reasoningMenuItem) {
      _reasoningMenuItem.textContent = `Reasoning: ${_showReasoning ? 'On' : 'Off'}`;
    }
  }

  function _loadReasoningPref() {
    try {
      const raw = window.localStorage.getItem(_REASONING_PREF_KEY);
      if (raw === null) {
        _setReasoningVisible(true);
        return;
      }
      _setReasoningVisible(raw === '1');
    } catch {
      _setReasoningVisible(true);
    }
  }

  function _saveReasoningPref() {
    try {
      window.localStorage.setItem(_REASONING_PREF_KEY, _showReasoning ? '1' : '0');
    } catch {
      // best effort
    }
  }

  function _applyTheme(theme) {
    _theme = theme === 'light' ? 'light' : 'dark';
    document.documentElement.dataset.theme = _theme;
    if (_themeMenuItem) {
      _themeMenuItem.textContent = `Theme: ${_theme === 'dark' ? 'Dark' : 'Light'}`;
    }
  }

  function _loadThemePref() {
    // The inline head script already applied the stored theme before first
    // paint; adopt that value so the menu label matches what is on screen.
    _applyTheme(document.documentElement.dataset.theme === 'light' ? 'light' : 'dark');
  }

  function _saveThemePref() {
    try {
      window.localStorage.setItem(_THEME_PREF_KEY, _theme);
    } catch {
      // best effort
    }
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
    const label = (panelEl.querySelector('h2')?.textContent || 'section').trim().toLowerCase();
    collapseBtnEl.title = `${collapsed ? 'Expand' : 'Collapse'} ${label} section`;
  }

  const _ACTIVE_PANEL_KEY = 'hearth:active_panel';

  function expandMusicPanel() {
    setPanelCollapsed(musicPanel, musicCollapseBtn, false);
    setPanelCollapsed(sessionsPanel, sessionsCollapseBtn, true);
    try { window.localStorage.setItem(_ACTIVE_PANEL_KEY, 'music'); } catch {}
  }

  function expandSessionsPanel() {
    setPanelCollapsed(sessionsPanel, sessionsCollapseBtn, false);
    setPanelCollapsed(musicPanel, musicCollapseBtn, true);
    try { window.localStorage.setItem(_ACTIVE_PANEL_KEY, 'sessions'); } catch {}
  }

  function _bindCollapsiblePanels() {
    sessionsCollapseBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      const willBeCollapsed = !sessionsPanel?.classList.contains('is-collapsed');
      if (!willBeCollapsed) {
        expandSessionsPanel();
      } else {
        setPanelCollapsed(sessionsPanel, sessionsCollapseBtn, true);
      }
    });

    musicCollapseBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      const willBeCollapsed = !musicPanel?.classList.contains('is-collapsed');
      if (!willBeCollapsed) {
        expandMusicPanel();
      } else {
        setPanelCollapsed(musicPanel, musicCollapseBtn, true);
      }
    });

    // Restore saved panel preference (default: sessions expanded, music collapsed)
    try {
      const saved = window.localStorage.getItem(_ACTIVE_PANEL_KEY);
      if (saved === 'music') {
        expandMusicPanel();
      } else {
        expandSessionsPanel();
      }
    } catch {
      expandSessionsPanel();
    }
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
    const outputMenu = document.getElementById('music-output-menu');
    if (sidebar?.contains(target) || sidebarToggleBtn?.contains(target) || outputMenu?.contains(target)) return;
    closeSidebar();
  });

  window.addEventListener('resize', () => {
    if (!isMobileLayout()) closeSidebar();
  });

  document.addEventListener('click', (e) => {
    if (!_sessionMenuEl || _sessionMenuEl.style.display === 'none') return;
    if (_sessionMenuEl.contains(e.target)) return;
    if (e.target.closest && e.target.closest('.session-kebab-btn')) return;
    closeSessionMenu();
  });

  document.addEventListener('click', (e) => {
    if (!_settingsMenuEl || _settingsMenuEl.style.display === 'none') return;
    if (_settingsMenuEl.contains(e.target)) return;
    if (authUsernameBtn?.contains(e.target)) return;
    closeSettingsMenu();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    closeSessionMenu();
    closeSettingsMenu();
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

  function setLocked(locked) {
    sendBtn.disabled = locked;
    input.disabled = locked;
    if (sessionNewBtn) sessionNewBtn.disabled = locked;
    if (memoryModalClearBtn) memoryModalClearBtn.disabled = locked;
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
      bubble.innerHTML = _renderMarkdownSafe(text || '');
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

  function appendModelBadge(wrapper, modelName, intent, fallback, tool) {
    if (!modelName) return;
    const isCloud = modelName.toLowerCase().includes('claude');
    wrapper.querySelector('.model-badge')?.remove();

    const badge = document.createElement('span');
    badge.className = 'model-badge ' + (isCloud ? 'cloud' : 'local');

    const rawTool = (tool || '').trim().toLowerCase();
    const rawIntent = (intent || '').trim().toLowerCase();

    // Determine clean human-friendly capability/tool label
    let cleanLabel = '';
    if (rawTool && rawTool !== 'none') {
      cleanLabel = rawTool;
    } else if (rawIntent === 'music' || rawIntent === 'weather' || rawIntent === 'code') {
      cleanLabel = rawIntent;
    } else if (rawIntent === 'code-question') {
      cleanLabel = 'code';
    } else if (rawIntent === 'memory-needed' || rawIntent === 'memory-augmented') {
      cleanLabel = 'memory';
    } else if (rawIntent === 'vision') {
      cleanLabel = 'vision';
    }

    let text = modelName;
    if (modelName.toLowerCase() === 'music') {
      text = 'music';
    } else if (cleanLabel && cleanLabel !== modelName.toLowerCase()) {
      text += ` · ${cleanLabel}`;
    }

    if (fallback) {
      text += ' · fallback';
    }

    badge.textContent = text;

    const tooltipLines = [
      `Model: ${modelName}`,
      tool ? `Tool: ${tool}` : null,
      intent ? `Intent: ${intent}` : null,
      `Route: ${isCloud ? 'cloud' : 'local'}${fallback ? ' (fallback)' : ''}`,
    ].filter(Boolean);
    badge.title = tooltipLines.join('\n');

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

  function appendReasoningSummary(wrapper, summary, routeType, plannerStatus) {
    const text = String(summary || '').trim();
    if (!text) return;

    wrapper.querySelector('.reasoning-block')?.remove();

    const details = document.createElement('details');
    details.className = 'reasoning-block';

    const summaryEl = document.createElement('summary');
    summaryEl.textContent = 'Reasoning summary';

    const contentEl = document.createElement('div');
    contentEl.className = 'reasoning-content';
    const meta = [];
    if (routeType) meta.push(`route=${routeType}`);
    if (plannerStatus) meta.push(`planner=${plannerStatus}`);
    contentEl.textContent = meta.length ? `${meta.join(' · ')}\n${text}` : text;

    details.appendChild(summaryEl);
    details.appendChild(contentEl);
    wrapper.appendChild(details);
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
    closeSessionMenu();
    sessionListEl.innerHTML = '';
    if (!sessions.length) {
      const div = document.createElement('div');
      div.className = 'list-item';
      div.innerHTML = '<div class="list-item-title">No sessions yet</div>';
      sessionListEl.appendChild(div);
      return;
    }

    for (const session of sessions) {
      const displayTitle = (session.title || session.preview || 'New session').slice(0, 80);
      const item = document.createElement('div');
      item.className = 'list-item session-list-item' + (session.session_id === activeId ? ' active' : '');
      item.innerHTML = `
        <div class="session-row">
          <div class="list-item-title session-title">${_esc(displayTitle)}</div>
          <button class="session-kebab-btn" data-sid="${session.session_id}" title="Session options" aria-label="Session options" aria-haspopup="menu">&#8942;</button>
        </div>
      `;
      item.addEventListener('click', () => selectSession(session.session_id));
      const kebab = item.querySelector('.session-kebab-btn');
      kebab.addEventListener('click', (e) => {
        e.stopPropagation();
        if (isSessionMenuOpenFor(session.session_id)) {
          closeSessionMenu();
        } else {
          openSessionMenu(kebab, session);
        }
      });
      sessionListEl.appendChild(item);
    }
  }

  function renderMemory(items) {
    const listEl = document.getElementById('memory-modal-list');
    const countEl = document.getElementById('memory-modal-count');
    if (countEl) countEl.textContent = `${items.length} item${items.length === 1 ? '' : 's'}`;
    if (!listEl) return;
    listEl.innerHTML = '';
    if (!items.length) {
      const div = document.createElement('div');
      div.className = 'memory-modal-item';
      div.innerHTML = '<div class="memory-modal-key" style="color:var(--text-muted);font-style:italic">No stored memory yet</div>';
      listEl.appendChild(div);
      return;
    }

    for (const item of items) {
      const tier = String(item.tier || '').toLowerCase();
      const tierLabel = tier === 'episodic' ? 'Episodic' : tier === 'semantic' ? 'Semantic' : 'Working';
      const consolidatedLabel = tier === 'episodic'
        ? (item.consolidated ? 'Consolidated' : 'Pending consolidation')
        : '';

      const div = document.createElement('div');
      div.className = 'memory-modal-item';
      div.innerHTML = `
        <div class="memory-modal-header">
          <span class="memory-modal-key">${_esc(item.key)}</span>
          <button class="panel-btn mini-btn danger memory-delete-btn" data-id="${_esc(item.id)}">Delete</button>
        </div>
        <div class="memory-modal-value">${_esc(item.value || '')}</div>
        <div class="memory-modal-meta">
          <span class="queue-badge">${_esc(tierLabel)}</span>
          ${consolidatedLabel ? `<span>· ${_esc(consolidatedLabel)}</span>` : ''}
        </div>
      `;
      div.querySelector('.memory-delete-btn')?.addEventListener('click', async (e) => {
        e.stopPropagation();
        await deleteMemory(item.id);
      });
      listEl.appendChild(div);
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
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function _safeHref(url) {
    const raw = String(url || '').trim();
    if (!raw) return '#';
    try {
      const parsed = new URL(raw, window.location.origin);
      const protocol = parsed.protocol.toLowerCase();
      if (protocol === 'http:' || protocol === 'https:' || protocol === 'mailto:' || protocol === 'tel:') {
        return parsed.href;
      }
    } catch {
      // fall through to safe default
    }
    return '#';
  }

  function _sanitizeMarkdownHtml(root) {
    const allowedTags = new Set([
      'A', 'P', 'BR', 'STRONG', 'EM', 'CODE', 'PRE', 'UL', 'OL', 'LI',
      'BLOCKQUOTE', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'HR'
    ]);

    const walk = (node) => {
      if (node.nodeType === Node.ELEMENT_NODE) {
        const tag = node.tagName;

        if (!allowedTags.has(tag)) {
          const replacement = document.createTextNode(node.textContent || '');
          node.replaceWith(replacement);
          return;
        }

        for (const attr of Array.from(node.attributes)) {
          const name = attr.name.toLowerCase();
          if (tag === 'A' && (name === 'href' || name === 'title')) {
            continue;
          }
          node.removeAttribute(attr.name);
        }

        if (tag === 'A') {
          node.setAttribute('href', _safeHref(node.getAttribute('href')));
          node.setAttribute('rel', 'noopener noreferrer nofollow ugc');
          node.setAttribute('target', '_blank');
        }
      }

      const children = Array.from(node.childNodes);
      for (const child of children) {
        walk(child);
      }
    };

    walk(root);
  }

  function _renderMarkdownSafe(text) {
    const rendered = marked.parse(text || '');
    const tpl = document.createElement('template');
    tpl.innerHTML = rendered;
    _sanitizeMarkdownHtml(tpl.content);
    return tpl.innerHTML;
  }

  let _currentDuration = 0;
  let _currentElapsed = 0;
  let _playbackState = 'stop';
  let _lastVolume = 60;
  let _progressTicker = null;

  function _formatTime(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return '0:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  }

  function _updateProgressDisplay(elapsed, duration) {
    const fill = document.getElementById('music-progress-fill');
    const elapsedEl = document.getElementById('music-time-elapsed');
    const durationEl = document.getElementById('music-time-duration');
    const safeDuration = Number.isFinite(duration) && duration > 0 ? duration : 0;
    const safeElapsed = Number.isFinite(elapsed) && elapsed >= 0 ? Math.min(elapsed, safeDuration || elapsed) : 0;
    if (elapsedEl) elapsedEl.textContent = _formatTime(safeElapsed);
    if (durationEl) durationEl.textContent = safeDuration > 0 ? _formatTime(safeDuration) : '0:00';
    if (fill) {
      const pct = safeDuration > 0 ? Math.min(100, (safeElapsed / safeDuration) * 100) : 0;
      fill.style.width = `${pct}%`;
    }
  }

  function _startProgressTicker() {
    if (_progressTicker) clearInterval(_progressTicker);
    _progressTicker = setInterval(() => {
      if (_playbackState === 'play' && _currentDuration > 0) {
        _currentElapsed = Math.min(_currentElapsed + 1, _currentDuration);
        _updateProgressDisplay(_currentElapsed, _currentDuration);
      }
    }, 1000);
  }

  function _stopProgressTicker() {
    if (_progressTicker) {
      clearInterval(_progressTicker);
      _progressTicker = null;
    }
  }

  async function refreshNowPlaying(autoExpand = false) {
    const titleEl = document.getElementById('music-track-title');
    const subtitleEl = document.getElementById('music-track-artist-album');
    const btn = document.getElementById('music-play-pause-btn');
    const volumeInput = document.getElementById('music-volume');
    const volumeValue = document.getElementById('music-volume-value');
    if (!titleEl) return;
    try {
      const resp = await (window.apiFetch || fetch)('/music/now_playing', { credentials: 'same-origin' });
      if (!resp.ok) return;
      const data = await resp.json();
      currentQueuePos = Number.isInteger(data.pos) ? data.pos : null;
      const previousState = _playbackState;
      _playbackState = data.state || 'stop';
      const isPlaying = data.track && data.state !== 'stop';

      if (isPlaying) {
        const t = data.track;
        const trackTitle = t.title || 'Unknown track';
        const subParts = [t.artist, t.album].filter(Boolean);
        const nowPlayingText = [t.artist, t.title].filter(Boolean).join(' — ') || trackTitle;

        titleEl.textContent = trackTitle;
        titleEl.classList.remove('now-playing-idle');
        if (subtitleEl) subtitleEl.textContent = subParts.join(' • ');
        if (musicCollapsedNowPlayingEl) {
          musicCollapsedNowPlayingEl.textContent = nowPlayingText;
          musicCollapsedNowPlayingEl.classList.remove('now-playing-idle');
        }
        if (btn) btn.textContent = data.state === 'play' ? '⏸' : '▶';

        _currentElapsed = Number.isFinite(data.elapsed) ? data.elapsed : 0;
        _currentDuration = Number.isFinite(data.duration) ? data.duration : 0;
        _updateProgressDisplay(_currentElapsed, _currentDuration);

        if (data.state === 'play') {
          _startProgressTicker();
          if (autoExpand || previousState !== 'play') {
            expandMusicPanel();
          }
          _ensureWebPlayerPlaying();
          _updateMediaSession(data.track, true);
        } else {
          _stopProgressTicker();
          if (_currentOutputTarget === 'phone' || _currentOutputTarget === 'both') {
            const webPlayer = document.getElementById('hearth-web-player');
            if (webPlayer && !webPlayer.paused) {
              webPlayer.pause();
            }
          }
          _updateMediaSession(data.track, false);
        }
      } else {
        _stopProgressTicker();
        if (_currentOutputTarget === 'phone' || _currentOutputTarget === 'both') {
          const webPlayer = document.getElementById('hearth-web-player');
          if (webPlayer && !webPlayer.paused) {
            webPlayer.pause();
          }
        }
        _updateMediaSession(null, false);
        titleEl.textContent = 'Nothing playing';
        titleEl.classList.add('now-playing-idle');
        if (subtitleEl) subtitleEl.textContent = '';
        if (musicCollapsedNowPlayingEl) {
          musicCollapsedNowPlayingEl.textContent = 'Nothing playing';
          musicCollapsedNowPlayingEl.classList.add('now-playing-idle');
        }
        if (btn) btn.textContent = '▶';
        _currentElapsed = 0;
        _currentDuration = 0;
        _updateProgressDisplay(0, 0);
      }

      if (volumeInput && Number.isFinite(data.volume)) {
        const vol = Math.max(0, Math.min(100, Number(data.volume)));
        volumeInput.value = String(vol);
        if (volumeValue) volumeValue.textContent = `${vol}%`;
        if (vol > 0) _lastVolume = vol;
      }
    } catch {
      // non-fatal — MPD may not be running
    }
  }

  async function refreshQueue() {
    const list = document.getElementById('queue-list');
    const countBadge = document.getElementById('music-queue-count');
    if (!list) return;
    try {
      const resp = await (window.apiFetch || fetch)('/music/queue', { credentials: 'same-origin' });
      if (!resp.ok) return;
      const data = await resp.json();
      const items = data.queue || [];
      if (countBadge) countBadge.textContent = String(items.length);
      if (!items.length) {
        list.innerHTML = '<div class="list-item" style="color:var(--text-muted);font-style:italic;font-size:0.75rem">Queue empty</div>';
        return;
      }
      list.innerHTML = items.map(item => {
        const active = currentQueuePos === item.pos ? ' active-track' : '';
        const title = item.title || 'Unknown track';
        const artist = [item.artist, item.album].filter(Boolean).join(' • ');
        return `<div class="list-item list-item-clickable${active}" data-pos="${item.pos}" title="Play this track">` +
               `<span class="queue-track-title">${_esc(title)}</span>` +
               (artist ? `<span class="queue-track-artist">${_esc(artist)}</span>` : '') +
               `</div>`;
      }).join('');
      list.querySelectorAll('.list-item-clickable').forEach(el => {
        el.addEventListener('click', () => {
          _ensureWebPlayerPlaying();
          musicControl('play_pos', { pos: parseInt(el.dataset.pos, 10) });
        });
      });
    } catch {
      // non-fatal
    }
  }

  let _currentOutputTarget = localStorage.getItem('hearth:music_output') || 'host';

  function _ensureWebPlayerPlaying() {
    if (_currentOutputTarget !== 'phone' && _currentOutputTarget !== 'both') return;
    const webPlayer = document.getElementById('hearth-web-player');
    if (!webPlayer) return;
    if (!webPlayer.src || webPlayer.src.indexOf('/music/stream') === -1) {
      webPlayer.src = '/music/stream?t=' + Date.now();
    }
    webPlayer.play().catch(() => {});
  }

  function _updateMediaSession(track, isPlaying) {
    if (!('mediaSession' in navigator)) return;
    try {
      if (track) {
        navigator.mediaSession.metadata = new MediaMetadata({
          title: track.title || 'Unknown Title',
          artist: track.artist || 'Unknown Artist',
          album: track.album || '',
        });
      }
      navigator.mediaSession.playbackState = isPlaying ? 'playing' : 'paused';
    } catch {
      // non-fatal
    }
  }

  function _updateOutputButtonUI(target) {
    const outputIcon = document.getElementById('music-output-icon');
    const outputLabel = document.getElementById('music-output-label');
    const outputBtn = document.getElementById('music-output-btn');
    if (!outputBtn) return;
    if (target === 'phone') {
      if (outputIcon) outputIcon.textContent = '📱';
      if (outputLabel) outputLabel.textContent = 'Device';
      outputBtn.classList.add('active-phone');
      outputBtn.title = 'Audio output: This device (phone/browser)';
    } else if (target === 'both') {
      if (outputIcon) outputIcon.textContent = '🌐';
      if (outputLabel) outputLabel.textContent = 'Both';
      outputBtn.classList.add('active-phone');
      outputBtn.title = 'Audio output: Host + This device';
    } else {
      if (outputIcon) outputIcon.textContent = '🖥️';
      if (outputLabel) outputLabel.textContent = 'Host';
      outputBtn.classList.remove('active-phone');
      outputBtn.title = 'Audio output: Host speakers';
    }
  }

  async function _switchOutputTarget(target) {
    _currentOutputTarget = target;
    localStorage.setItem('hearth:music_output', target);
    _updateOutputButtonUI(target);

    const webPlayer = document.getElementById('hearth-web-player');

    if (webPlayer && target === 'host') {
      webPlayer.pause();
      webPlayer.removeAttribute('src');
      webPlayer.load();
    }

    try {
      if (target === 'phone') {
        // Output 1 = Web Stream (exclusive)
        await (window.apiFetch || fetch)('/music/outputs/select', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({ output_id: '1', mode: 'exclusive' }),
        });
      } else if (target === 'both') {
        // Output 0 = Host, Output 1 = Web Stream (mirror)
        await (window.apiFetch || fetch)('/music/outputs/select', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({ output_id: '0', mode: 'enable' }),
        });
        await (window.apiFetch || fetch)('/music/outputs/select', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({ output_id: '1', mode: 'enable' }),
        });
      } else {
        // Output 0 = Host (exclusive)
        await (window.apiFetch || fetch)('/music/outputs/select', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({ output_id: '0', mode: 'exclusive' }),
        });
      }
    } catch {
      // non-fatal
    }

    if (target === 'phone' || target === 'both') {
      _ensureWebPlayerPlaying();
    }
  }

  function _bindMusicOutputs() {
    const outputBtn = document.getElementById('music-output-btn');
    const outputMenu = document.getElementById('music-output-menu');
    const outputOptions = document.getElementById('music-output-options');
    if (!outputBtn || !outputMenu) return;

    if (outputMenu.parentElement !== document.body) {
      document.body.appendChild(outputMenu);
    }

    _updateOutputButtonUI(_currentOutputTarget);

    const webPlayer = document.getElementById('hearth-web-player');
    if (webPlayer) {
      webPlayer.addEventListener('error', () => {
        if (_currentOutputTarget === 'phone' || _currentOutputTarget === 'both') {
          setTimeout(() => {
            _ensureWebPlayerPlaying();
          }, 800);
        }
      });
    }

    function _positionMenu() {
      const rect = outputBtn.getBoundingClientRect();
      const margin = 8;
      const menuWidth = Math.min(230, window.innerWidth - margin * 2);
      outputMenu.style.width = `${menuWidth}px`;

      let left = rect.left;
      if (left + menuWidth > window.innerWidth - margin) {
        left = Math.max(margin, window.innerWidth - menuWidth - margin);
      }
      outputMenu.style.left = `${Math.round(left)}px`;

      const menuHeight = outputMenu.offsetHeight || 135;
      let top;
      if (rect.top - menuHeight - margin >= margin) {
        top = rect.top - menuHeight - margin;
      } else if (rect.bottom + margin + menuHeight <= window.innerHeight - margin) {
        top = rect.bottom + margin;
      } else {
        top = Math.max(margin, Math.min(window.innerHeight - menuHeight - margin, rect.top - menuHeight - margin));
      }
      outputMenu.style.top = `${Math.round(top)}px`;
    }

    outputBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const isHidden = outputMenu.classList.contains('hidden');
      if (isHidden) {
        _renderOutputOptions();
        outputMenu.classList.remove('hidden');
        _positionMenu();
        outputBtn.setAttribute('aria-expanded', 'true');
      } else {
        outputMenu.classList.add('hidden');
        outputBtn.setAttribute('aria-expanded', 'false');
      }
    });

    document.addEventListener('click', (e) => {
      if (outputMenu.classList.contains('hidden')) return;
      if (outputMenu.contains(e.target) || outputBtn.contains(e.target)) return;
      outputMenu.classList.add('hidden');
      outputBtn.setAttribute('aria-expanded', 'false');
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !outputMenu.classList.contains('hidden')) {
        outputMenu.classList.add('hidden');
        outputBtn.setAttribute('aria-expanded', 'false');
      }
    });

    window.addEventListener('resize', () => {
      if (!outputMenu.classList.contains('hidden')) {
        _positionMenu();
      }
    });

    window.addEventListener('scroll', () => {
      if (!outputMenu.classList.contains('hidden')) {
        _positionMenu();
      }
    }, true);

    function _renderOutputOptions() {
      if (!outputOptions) return;
      const options = [
        { id: 'host', label: '🖥️ Host Speakers', desc: 'Computer speakers' },
        { id: 'phone', label: '📱 This Device', desc: 'Play on this phone/browser' },
        { id: 'both', label: '🌐 Everywhere', desc: 'Host + this device' },
      ];
      outputOptions.innerHTML = options.map((opt) => {
        const isSelected = _currentOutputTarget === opt.id;
        const check = isSelected ? '<span style="color:var(--accent);font-weight:bold;margin-left:0.5rem;">✓</span>' : '';
        return (
          `<button type="button" class="music-output-option${isSelected ? ' selected' : ''}" data-target="${opt.id}">` +
          `<div style="display:flex;flex-direction:column;gap:0.1rem;text-align:left;">` +
          `<span style="font-weight:500;">${opt.label}</span>` +
          `<span style="font-size:0.7rem;color:var(--text-muted);">${opt.desc}</span>` +
          `</div>` +
          `${check}` +
          `</button>`
        );
      }).join('');

      outputOptions.querySelectorAll('.music-output-option').forEach((btn) => {
        btn.addEventListener('click', async (e) => {
          e.stopPropagation();
          const target = btn.dataset.target;
          outputMenu.classList.add('hidden');
          outputBtn.setAttribute('aria-expanded', 'false');
          await _switchOutputTarget(target);
        });
      });
    }

    if ('mediaSession' in navigator) {
      try {
        navigator.mediaSession.setActionHandler('play', () => musicControl('resume'));
        navigator.mediaSession.setActionHandler('pause', () => musicControl('pause'));
        navigator.mediaSession.setActionHandler('previoustrack', () => musicControl('previous'));
        navigator.mediaSession.setActionHandler('nexttrack', () => musicControl('next'));
        navigator.mediaSession.setActionHandler('stop', () => musicControl('stop'));
      } catch {
        // non-fatal
      }
    }
  }

  async function musicControl(action, extra = {}) {
    if (action === 'pause' || action === 'stop') {
      const webPlayer = document.getElementById('hearth-web-player');
      if (webPlayer && !webPlayer.paused) {
        webPlayer.pause();
      }
    }
    try {
      await (window.apiFetch || fetch)('/music/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ action, ...extra }),
      });
      // Brief delay so MPD state settles before polling.
      const shouldAutoExpand = action === 'resume' || action === 'play_pos';
      setTimeout(() => { refreshNowPlaying(shouldAutoExpand); refreshQueue(); }, 400);
    } catch {
      // non-fatal
    }
  }

  // Wire music control buttons.
  (function _bindMusicControls() {
    const pp = document.getElementById('music-play-pause-btn');
    const prev = document.getElementById('music-prev-btn');
    const next = document.getElementById('music-next-btn');
    const stop = document.getElementById('music-stop-btn');
    const shuffle = document.getElementById('music-shuffle-btn');
    const clearBtn = document.getElementById('music-queue-clear-btn');
    const muteBtn = document.getElementById('music-mute-btn');
    const volume = document.getElementById('music-volume');
    const volumeValue = document.getElementById('music-volume-value');

    _bindMusicOutputs();

    if (pp) pp.addEventListener('click', async () => {
      // Toggle based on current label (▶ = resume, ⏸ = pause).
      const action = pp.textContent.trim() === '⏸' ? 'pause' : 'resume';
      if (action === 'resume') {
        _ensureWebPlayerPlaying();
      }
      await musicControl(action);
    });
    if (prev) prev.addEventListener('click', () => musicControl('previous'));
    if (next) next.addEventListener('click', () => musicControl('next'));
    if (stop) stop.addEventListener('click', () => musicControl('stop'));
    if (shuffle) shuffle.addEventListener('click', () => musicControl('shuffle'));
    if (clearBtn) clearBtn.addEventListener('click', () => musicControl('clear'));
    if (muteBtn) {
      muteBtn.addEventListener('click', async () => {
        const currentVol = volume ? parseInt(volume.value, 10) || 0 : 0;
        if (currentVol > 0) {
          _lastVolume = currentVol;
          await musicControl('set_volume', { volume: 0 });
        } else {
          await musicControl('set_volume', { volume: _lastVolume || 60 });
        }
      });
    }
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

  function _ensureSessionMenu() {
    if (_sessionMenuEl) return _sessionMenuEl;
    const el = document.createElement('div');
    el.className = 'session-menu';
    el.setAttribute('role', 'menu');
    el.style.display = 'none';
    el.innerHTML =
      '<button type="button" class="session-menu-item" data-action="rename" role="menuitem">Rename</button>' +
      '<button type="button" class="session-menu-item danger" data-action="delete" role="menuitem">Delete</button>';
    el.querySelector('[data-action="rename"]').addEventListener('click', () => {
      const sid = _sessionMenuSid;
      const prev = _sessionMenuTitle;
      closeSessionMenu();
      if (sid) renameSession(sid, prev);
    });
    el.querySelector('[data-action="delete"]').addEventListener('click', () => {
      const sid = _sessionMenuSid;
      closeSessionMenu();
      if (sid) deleteSession(sid);
    });
    document.body.appendChild(el);
    _sessionMenuEl = el;
    return el;
  }

  function isSessionMenuOpenFor(sid) {
    return !!_sessionMenuEl && _sessionMenuEl.style.display !== 'none' && _sessionMenuSid === sid;
  }

  function openSessionMenu(kebabBtn, session) {
    const menu = _ensureSessionMenu();
    _sessionMenuSid = session.session_id;
    _sessionMenuTitle = session.title || '';
    menu.style.display = 'flex';
    const rect = kebabBtn.getBoundingClientRect();
    const margin = 4;
    menu.style.left = 'auto';
    menu.style.right = (window.innerWidth - rect.right) + 'px';
    menu.style.top = (rect.bottom + margin) + 'px';
    const mrect = menu.getBoundingClientRect();
    if (mrect.left < margin) {
      menu.style.right = 'auto';
      menu.style.left = Math.max(margin, rect.left - mrect.width + kebabBtn.offsetWidth) + 'px';
    }
    const mrect2 = menu.getBoundingClientRect();
    if (mrect2.bottom > window.innerHeight - margin) {
      menu.style.top = (rect.top - mrect2.height - margin) + 'px';
    }
  }

  function closeSessionMenu() {
    if (_sessionMenuEl) _sessionMenuEl.style.display = 'none';
    _sessionMenuSid = null;
    _sessionMenuTitle = '';
  }

  // ── Settings menu (opened from the username in the sidebar) ─────────────────

  function _ensureSettingsMenu() {
    if (_settingsMenuEl) return _settingsMenuEl;
    const el = document.createElement('div');
    el.className = 'settings-menu';
    el.setAttribute('role', 'menu');
    el.style.display = 'none';
    el.innerHTML =
      '<button type="button" class="settings-menu-item" data-action="theme" role="menuitem">Theme: Dark</button>' +
      '<button type="button" class="settings-menu-item" data-action="reasoning" role="menuitem">Reasoning: On</button>' +
      '<div class="settings-menu-sep" role="separator"></div>' +
      '<button type="button" class="settings-menu-item" data-action="memory" role="menuitem">Manage memory</button>' +
      '<button type="button" class="settings-menu-item" data-action="beets" role="menuitem">Update music library</button>' +
      '<div class="settings-menu-sep" role="separator"></div>' +
      '<button type="button" class="settings-menu-item danger" data-action="logout" role="menuitem">Sign out</button>';
    _themeMenuItem = el.querySelector('[data-action="theme"]');
    _reasoningMenuItem = el.querySelector('[data-action="reasoning"]');
    _beetsMenuItem = el.querySelector('[data-action="beets"]');
    _consolidateMenuItem = null;
    _themeMenuItem.addEventListener('click', () => {
      _applyTheme(_theme === 'dark' ? 'light' : 'dark');
      _saveThemePref();
    });
    _reasoningMenuItem.addEventListener('click', () => {
      _setReasoningVisible(!_showReasoning);
      _saveReasoningPref();
    });
    el.querySelector('[data-action="memory"]').addEventListener('click', () => {
      openMemoryModal();
    });
    _beetsMenuItem.addEventListener('click', () => { void updateMusicLibrary(); });
    el.querySelector('[data-action="logout"]').addEventListener('click', () => {
      closeSettingsMenu();
      if (typeof window.hearthLogout === 'function') {
        window.hearthLogout();
      }
    });
    document.body.appendChild(el);
    _settingsMenuEl = el;
    return el;
  }

  function openMemoryModal() {
    closeSettingsMenu();
    if (memoryModal) {
      memoryModal.style.display = 'flex';
      void refreshMemory();
    }
  }

  function closeMemoryModal() {
    if (memoryModal) memoryModal.style.display = 'none';
  }

  (function _bindMemoryModal() {
    memoryModalCloseBtn?.addEventListener('click', closeMemoryModal);
    memoryModalConsolidateBtn?.addEventListener('click', () => { void consolidateMemoryNow(); });
    memoryModalClearBtn?.addEventListener('click', () => { void clearAllMemory(); });

    memoryModal?.addEventListener('click', (e) => {
      if (e.target === memoryModal) closeMemoryModal();
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && memoryModal && memoryModal.style.display !== 'none') {
        closeMemoryModal();
      }
    });
  })();

  function openSettingsMenu(trigger) {
    const menu = _ensureSettingsMenu();
    _themeMenuItem.textContent = `Theme: ${_theme === 'dark' ? 'Dark' : 'Light'}`;
    _reasoningMenuItem.textContent = `Reasoning: ${_showReasoning ? 'On' : 'Off'}`;
    menu.style.display = 'flex';
    const rect = trigger.getBoundingClientRect();
    const margin = 4;
    menu.style.left = 'auto';
    menu.style.right = (window.innerWidth - rect.right) + 'px';
    menu.style.top = (rect.bottom + margin) + 'px';
    const mrect = menu.getBoundingClientRect();
    if (mrect.left < margin) {
      menu.style.right = 'auto';
      menu.style.left = Math.max(margin, rect.left - mrect.width) + 'px';
    }
    const mrect2 = menu.getBoundingClientRect();
    if (mrect2.bottom > window.innerHeight - margin) {
      menu.style.top = (rect.top - mrect2.height - margin) + 'px';
    }
    authUsernameBtn?.setAttribute('aria-expanded', 'true');
  }

  function closeSettingsMenu() {
    if (_settingsMenuEl) _settingsMenuEl.style.display = 'none';
    authUsernameBtn?.setAttribute('aria-expanded', 'false');
  }

  authUsernameBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    if (_settingsMenuEl && _settingsMenuEl.style.display !== 'none') {
      closeSettingsMenu();
    } else {
      openSettingsMenu(authUsernameBtn);
    }
  });

  function _setSettingsActionsBusy(busy) {
    _settingsActionBusy = busy;
    if (_beetsMenuItem) _beetsMenuItem.disabled = busy;
    if (_consolidateMenuItem) _consolidateMenuItem.disabled = busy;
  }

  async function updateMusicLibrary() {
    if (_settingsActionBusy) return;
    _setSettingsActionsBusy(true);
    if (_beetsMenuItem) _beetsMenuItem.textContent = 'Updating music…';
    try {
      const resp = await (window.apiFetch || fetch)('/music/beets/update', {
        method: 'POST',
        credentials: 'same-origin',
      });
      let data = null;
      try { data = await resp.json(); } catch { /* best effort */ }
      if (!resp.ok) throw new Error(data?.error || `HTTP ${resp.status}`);
      appendMessage('assistant', 'Music library updated.');
    } catch (err) {
      appendMessage('assistant', `⚠ Unable to update music library: ${err.message}`);
    } finally {
      _setSettingsActionsBusy(false);
      if (_beetsMenuItem) _beetsMenuItem.textContent = 'Update music library';
    }
  }

  async function consolidateMemoryNow() {
    if (_settingsActionBusy) return;
    _setSettingsActionsBusy(true);
    if (memoryModalConsolidateBtn) {
      memoryModalConsolidateBtn.textContent = 'Consolidating…';
      memoryModalConsolidateBtn.disabled = true;
    }
    try {
      const resp = await (window.apiFetch || fetch)('/memory/consolidate', {
        method: 'POST',
        credentials: 'same-origin',
      });
      let data = null;
      try { data = await resp.json(); } catch { /* best effort */ }
      if (!resp.ok) throw new Error(data?.error || `HTTP ${resp.status}`);
      await refreshMemory();
      appendMessage('assistant', 'Memory consolidated.');
    } catch (err) {
      appendMessage('assistant', `⚠ Unable to consolidate memory: ${err.message}`);
    } finally {
      _setSettingsActionsBusy(false);
      if (memoryModalConsolidateBtn) {
        memoryModalConsolidateBtn.textContent = 'Consolidate memory';
        memoryModalConsolidateBtn.disabled = false;
      }
    }
  }

  async function renameSession(sessionId, currentTitle) {
    const entered = prompt('Rename session:', currentTitle || '');
    if (entered === null) return;
    const title = entered.trim();
    try {
      const resp = await (window.apiFetch || fetch)(`/chat/sessions/${encodeURIComponent(sessionId)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ title }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      await refreshSessions();
    } catch (err) {
      appendMessage('assistant', `⚠ Unable to rename session: ${err.message}`);
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
    let reasoningAccumulated = '';
    let voiceMeta = null;
    let metaRouteType = '';
    let metaPlannerStatus = '';
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
      let currentModel = '';
      let currentIntent = '';
      let currentFallback = false;
      let currentTool = '';

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
            currentModel = parsed.model;
            if (parsed.intent !== undefined) currentIntent = parsed.intent;
            if (parsed.fallback !== undefined) currentFallback = Boolean(parsed.fallback);
            if (parsed.tool !== undefined) currentTool = parsed.tool || '';
            appendModelBadge(wrapper, currentModel, currentIntent, currentFallback, currentTool);
          }

          if (parsed.tool) {
            currentTool = parsed.tool;
            if (currentModel) {
              appendModelBadge(wrapper, currentModel, currentIntent, currentFallback, currentTool);
            }
          }

          if (parsed.route_type) {
            metaRouteType = String(parsed.route_type || '');
          }

          if (parsed.planner_status) {
            metaPlannerStatus = String(parsed.planner_status || '');
          }

          if (parsed.reasoning_summary) {
            appendReasoningSummary(
              wrapper,
              parsed.reasoning_summary,
              metaRouteType,
              metaPlannerStatus,
            );
          }

          if (parsed.thinking) {
            reasoningAccumulated += String(parsed.thinking || '');
            appendReasoningSummary(
              wrapper,
              reasoningAccumulated,
              metaRouteType,
              metaPlannerStatus,
            );
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
            bubble.innerHTML = _renderMarkdownSafe(accumulated);
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
      const isMusicMsg = /play|queue|music|song|track|artist|jazz|rock|classical/i.test(text || '') ||
                         /playing|added|resumed/i.test(accumulated || '');
      refreshNowPlaying(isMusicMsg);
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
    _loadReasoningPref();
    _loadThemePref();
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
