(() => {
  const messagesEl = document.getElementById('messages');
  const messagesInner = document.getElementById('messages-inner');
  const input = document.getElementById('message-input');
  const sendBtn = document.getElementById('send-btn');
  const stopBtn = document.getElementById('stop-btn');
  const sessionListEl = document.getElementById('session-list');
  const sessionNewBtn = document.getElementById('session-new-btn');
  const sessionsPanel = document.getElementById('sessions-panel');
  const projectsPanel = document.getElementById('projects-panel');
  const sidebarSectionChatsBtn = document.getElementById('sidebar-section-chats');
  const sidebarSectionProjectsBtn = document.getElementById('sidebar-section-projects');
  const projectListEl = document.getElementById('project-list');
  const projectNewBtn = document.getElementById('project-new-btn');
  const projectCreateForm = document.getElementById('project-create-form');
  const projectNameInput = document.getElementById('project-name-input');
  const projectFolderInput = document.getElementById('project-folder-input');
  const projectDescriptionInput = document.getElementById('project-description-input');
  const projectGitInitInput = document.getElementById('project-git-init-input');
  const projectCreateCancelBtn = document.getElementById('project-create-cancel-btn');
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
  const projectHeader = document.getElementById('project-header');
  const projectBackBtn = document.getElementById('project-back-btn');
  const projectTitleEl = document.getElementById('project-title');
  const projectModelIndicatorEl = document.getElementById('project-model-indicator');
  const projectReindexBtn = document.getElementById('project-reindex-btn');
  const projectFilesPane = document.getElementById('project-files-pane');
  const projectFilesListEl = document.getElementById('project-files-list');
  const projectIndexStatusEl = document.getElementById('project-index-status');
  const projectFileViewerEl = document.getElementById('project-file-viewer');
  const projectFileViewerPathEl = document.getElementById('project-file-viewer-path');
  const projectFileViewerContentEl = document.getElementById('project-file-viewer-content');

  window.appUi = { messagesEl, messagesInner, input, sendBtn };
  let currentSessionId = null;
  let currentSidebarSection = 'chats';
  let currentProject = null;
  let projectIndexPollTimer = null;
  let coderModelName = '';
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

  function _slugifyProjectName(text) {
    return String(text || '')
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 64);
  }

  function _encodePath(path) {
    return String(path || '')
      .split('/')
      .filter(Boolean)
      .map(encodeURIComponent)
      .join('/');
  }

  function _withProjectParam(path, projectId = currentProject?.id || '') {
    const pid = String(projectId || '').trim();
    if (!pid) return path;
    const joiner = path.includes('?') ? '&' : '?';
    return `${path}${joiner}project_id=${encodeURIComponent(pid)}`;
  }

  function _formatOpenedAt(ts) {
    if (!ts) return 'never opened';
    const deltaSec = Math.max(0, Math.floor(Date.now() / 1000 - Number(ts)));
    if (deltaSec < 60) return 'opened now';
    const minutes = Math.floor(deltaSec / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days < 7) return `${days}d ago`;
    const weeks = Math.floor(days / 7);
    return `${weeks}w ago`;
  }

  function _setSidebarSection(section) {
    const target = section === 'projects' ? 'projects' : 'chats';
    currentSidebarSection = target;
    const inProjects = target === 'projects';
    sidebarSectionChatsBtn?.classList.toggle('active', !inProjects);
    sidebarSectionChatsBtn?.setAttribute('aria-pressed', String(!inProjects));
    sidebarSectionProjectsBtn?.classList.toggle('active', inProjects);
    sidebarSectionProjectsBtn?.setAttribute('aria-pressed', String(inProjects));
    if (projectsPanel) projectsPanel.style.display = inProjects ? '' : 'none';
    if (sessionsPanel) sessionsPanel.style.display = inProjects ? 'none' : '';
    if (musicPanel) musicPanel.style.display = inProjects ? 'none' : '';
    if (memoryPanel) memoryPanel.style.display = inProjects ? 'none' : '';
  }

  function _setProjectWorkspaceActive(active) {
    if (projectHeader) projectHeader.style.display = active ? 'flex' : 'none';
    if (projectFilesPane) projectFilesPane.style.display = active ? 'flex' : 'none';
    if (!active && projectIndexStatusEl) projectIndexStatusEl.style.display = 'none';
    if (!active && projectFileViewerEl) projectFileViewerEl.style.display = 'none';
    if (!active) {
      if (projectIndexPollTimer) {
        clearInterval(projectIndexPollTimer);
        projectIndexPollTimer = null;
      }
      currentProject = null;
      resetMessagesView();
    }
  }

  function _setProjectIndexStatus(text, level = 'info') {
    if (!projectIndexStatusEl) return;
    projectIndexStatusEl.textContent = text;
    projectIndexStatusEl.style.display = text ? 'block' : 'none';
    projectIndexStatusEl.style.color = level === 'error' ? '#ff8787' : 'var(--text-muted)';
  }

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
  sidebarSectionChatsBtn?.addEventListener('click', () => {
    if (currentProject?.id) {
      backToChat();
      return;
    }
    _setSidebarSection('chats');
  });
  sidebarSectionProjectsBtn?.addEventListener('click', () => {
    _setSidebarSection('projects');
    refreshProjects();
  });
  projectBackBtn?.addEventListener('click', backToChat);
  projectReindexBtn?.addEventListener('click', triggerProjectReindex);
  projectNewBtn?.addEventListener('click', () => toggleProjectCreateForm(projectCreateForm?.style.display === 'none'));
  projectCreateCancelBtn?.addEventListener('click', () => toggleProjectCreateForm(false));
  projectNameInput?.addEventListener('input', () => {
    if (!projectFolderInput || projectFolderInput.dataset.manual === 'true') return;
    projectFolderInput.value = _slugifyProjectName(projectNameInput.value);
  });
  projectFolderInput?.addEventListener('input', () => {
    if (!projectFolderInput) return;
    projectFolderInput.dataset.manual = projectFolderInput.value.trim() ? 'true' : '';
  });
  projectCreateForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = projectNameInput?.value?.trim() || '';
    const folderName = (projectFolderInput?.value?.trim() || _slugifyProjectName(name)).slice(0, 120);
    if (!name || !folderName) return;
    try {
      const resp = await (window.apiFetch || fetch)('/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({
          name,
          folder_name: folderName,
          description: projectDescriptionInput?.value?.trim() || '',
          git_init: Boolean(projectGitInitInput?.checked),
        }),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${resp.status}`);
      }
      const project = await resp.json();
      toggleProjectCreateForm(false);
      await refreshProjects();
      await openProject(project.id);
    } catch (err) {
      appendMessage('assistant', `⚠ Unable to create project: ${err.message}`);
    }
  });

  function setLocked(locked) {
    sendBtn.disabled = locked;
    input.disabled = locked;
    if (sessionNewBtn) sessionNewBtn.disabled = locked;
    if (memoryClearBtn) memoryClearBtn.disabled = locked;
    if (projectNewBtn) projectNewBtn.disabled = locked;
    if (projectReindexBtn) projectReindexBtn.disabled = locked;
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

  function renderProjects(projects) {
    if (!projectListEl) return;
    projectListEl.innerHTML = '';
    if (!projects.length) {
      const div = document.createElement('div');
      div.className = 'list-item';
      div.innerHTML = '<div class="list-item-title">No projects yet</div>';
      projectListEl.appendChild(div);
      return;
    }

    for (const project of projects) {
      const item = document.createElement('div');
      const active = currentProject?.id === project.id;
      item.className = 'list-item' + (active ? ' active' : '');
      const gitBadge = project.git ? '<span class="project-git-badge">git</span>' : '';
      item.innerHTML = `
        <div class="project-file-item">
          <div class="list-item-title session-title">${_esc(project.name || 'Untitled')}</div>
          ${gitBadge}
        </div>
        <div class="list-item-meta">${_esc(_formatOpenedAt(project.opened_at))}</div>
      `;
      item.addEventListener('click', () => openProject(project.id));
      projectListEl.appendChild(item);
    }
  }

  async function refreshProjects() {
    try {
      const resp = await (window.apiFetch || fetch)('/projects', { credentials: 'same-origin' });
      if (!resp.ok) return;
      const data = await resp.json();
      renderProjects(data.projects || []);
    } catch {
      // non-fatal
    }
  }

  async function refreshProjectConfig() {
    try {
      const resp = await (window.apiFetch || fetch)('/projects/config', { credentials: 'same-origin' });
      if (!resp.ok) return;
      const data = await resp.json();
      coderModelName = String(data.coder_model || '').trim();
    } catch {
      // non-fatal
    }
  }

  function renderProjectFiles(files = []) {
    if (!projectFilesListEl) return;
    projectFilesListEl.innerHTML = '';
    if (!files.length) {
      const row = document.createElement('div');
      row.className = 'list-item';
      row.innerHTML = '<div class="list-item-title">No files found</div>';
      projectFilesListEl.appendChild(row);
      return;
    }
    for (const relPath of files) {
      const row = document.createElement('div');
      row.className = 'list-item';
      row.innerHTML = `<div class="list-item-title">${_esc(relPath)}</div>`;
      row.addEventListener('click', () => openProjectFile(relPath));
      projectFilesListEl.appendChild(row);
    }
  }

  async function openProjectFile(projectRelativePath) {
    if (!currentProject?.folder_name) return;
    const fullPath = `${String(currentProject.folder_name).replace(/\/+$/, '')}/${String(projectRelativePath).replace(/^\/+/, '')}`;
    try {
      const resp = await (window.apiFetch || fetch)(`/code/files/${_encodePath(fullPath)}`, { credentials: 'same-origin' });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      if (projectFileViewerPathEl) projectFileViewerPathEl.textContent = data.path || fullPath;
      if (projectFileViewerContentEl) projectFileViewerContentEl.textContent = data.content || '';
      if (projectFileViewerEl) projectFileViewerEl.style.display = 'flex';
    } catch (err) {
      appendMessage('assistant', `⚠ Unable to open file: ${err.message}`);
    }
  }

  async function refreshProjectIndexStatus() {
    if (!currentProject?.id) return;
    try {
      const resp = await (window.apiFetch || fetch)(`/projects/${encodeURIComponent(currentProject.id)}/index/status`, { credentials: 'same-origin' });
      if (!resp.ok) return;
      const data = await resp.json();
      if (data.status === 'running') {
        _setProjectIndexStatus('Indexing…');
      } else if (data.status === 'done') {
        const details = `Indexed ${data.files_indexed || 0} files · ${data.chunks || 0} chunks`;
        _setProjectIndexStatus(`Index ready · ${details}`);
        if (projectIndexPollTimer) {
          clearInterval(projectIndexPollTimer);
          projectIndexPollTimer = null;
        }
      } else if (data.status === 'error') {
        _setProjectIndexStatus(`Index error: ${data.error || 'unknown error'}`, 'error');
        if (projectIndexPollTimer) {
          clearInterval(projectIndexPollTimer);
          projectIndexPollTimer = null;
        }
      } else {
        _setProjectIndexStatus('Index idle');
      }
    } catch {
      // non-fatal
    }
  }

  function startProjectIndexPolling() {
    if (projectIndexPollTimer) clearInterval(projectIndexPollTimer);
    projectIndexPollTimer = setInterval(() => { void refreshProjectIndexStatus(); }, 1500);
    void refreshProjectIndexStatus();
  }

  async function triggerProjectReindex() {
    if (!currentProject?.id) return;
    try {
      const resp = await (window.apiFetch || fetch)(`/projects/${encodeURIComponent(currentProject.id)}/index`, {
        method: 'POST',
        credentials: 'same-origin',
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      _setProjectIndexStatus('Indexing…');
      startProjectIndexPolling();
    } catch (err) {
      _setProjectIndexStatus(`Unable to re-index: ${err.message}`, 'error');
    }
  }

  async function ensureProjectIndexStartedOnOpen() {
    if (!currentProject?.id) return;
    try {
      const statusResp = await (window.apiFetch || fetch)(`/projects/${encodeURIComponent(currentProject.id)}/index/status`, {
        credentials: 'same-origin',
      });
      if (!statusResp.ok) return;
      const status = await statusResp.json();
      if (status.status !== 'idle') return;

      const startResp = await (window.apiFetch || fetch)(`/projects/${encodeURIComponent(currentProject.id)}/index`, {
        method: 'POST',
        credentials: 'same-origin',
      });
      if (!startResp.ok) {
        throw new Error(`HTTP ${startResp.status}`);
      }
      _setProjectIndexStatus('Indexing…');
    } catch (err) {
      _setProjectIndexStatus(`Unable to start index: ${err.message}`, 'error');
    }
  }

  async function openProject(projectId) {
    closeSidebar();
    try {
      const resp = await (window.apiFetch || fetch)(`/projects/${encodeURIComponent(projectId)}/open`, {
        method: 'POST',
        credentials: 'same-origin',
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const payload = await resp.json();
      currentProject = payload.project || null;
      _setSidebarSection('projects');
      _setProjectWorkspaceActive(true);
      projectTitleEl.textContent = currentProject?.name || 'Project';
      projectModelIndicatorEl.textContent = coderModelName ? `Coder model: ${coderModelName}` : 'Coder model: unavailable';
      const filesResp = await (window.apiFetch || fetch)(`/projects/${encodeURIComponent(projectId)}/files`, { credentials: 'same-origin' });
      if (filesResp.ok) {
        const filesPayload = await filesResp.json();
        renderProjectFiles(filesPayload.files || []);
      } else {
        renderProjectFiles(payload.files || []);
      }
      await Promise.all([refreshSessions(), loadCurrentSessionMessages(), refreshProjects()]);
      await ensureProjectIndexStartedOnOpen();
      startProjectIndexPolling();
    } catch (err) {
      appendMessage('assistant', `⚠ Unable to open project: ${err.message}`);
    }
  }

  function backToChat() {
    _setProjectWorkspaceActive(false);
    _setSidebarSection('chats');
    void Promise.all([refreshSessions(), loadCurrentSessionMessages()]);
  }

  function toggleProjectCreateForm(show) {
    if (!projectCreateForm) return;
    projectCreateForm.style.display = show ? 'flex' : 'none';
    if (!show) {
      projectCreateForm.reset();
      return;
    }
    projectNameInput?.focus();
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
      const resp = await (window.apiFetch || fetch)(_withProjectParam('/chat/sessions'), { credentials: 'same-origin' });
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
      const resp = await (window.apiFetch || fetch)(_withProjectParam('/chat/session/messages'), { credentials: 'same-origin' });
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
      const resp = await (window.apiFetch || fetch)(_withProjectParam('/chat/session/select'), {
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
      const resp = await (window.apiFetch || fetch)(_withProjectParam(`/chat/sessions/${encodeURIComponent(sessionId)}`), {
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
          ...(currentProject?.id && { project_id: currentProject.id }),
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
    _setSidebarSection('chats');
    await Promise.all([refreshSessions(), refreshMemory(), refreshProjects(), refreshProjectConfig()]);
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
