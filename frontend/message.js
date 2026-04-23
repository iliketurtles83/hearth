  const messagesEl   = document.getElementById('messages');
    const messagesInner = document.getElementById('messages-inner');
    const input      = document.getElementById('message-input');
    const sendBtn    = document.getElementById('send-btn');
    const newChatBtn = document.getElementById('new-chat-btn');

    // Auto-grow textarea
    input.addEventListener('input', () => {
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 160) + 'px';
    });

    // Enter to send, Shift+Enter for newline
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
    });

    sendBtn.addEventListener('click', send);
    newChatBtn.addEventListener('click', startNewChat);

    // ── Helpers ────────────────────────────────────────────────

    function setLocked(locked) {
      sendBtn.disabled = locked;
      input.disabled   = locked;
      newChatBtn.disabled = locked;
    }

    function scrollToBottom() {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function appendMessage(role, text = '') {
      document.getElementById('empty-state')?.remove();

      const wrapper = document.createElement('div');
      wrapper.className = `message ${role}`;

      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.textContent = text;

      wrapper.appendChild(bubble);
      messagesInner.appendChild(wrapper);
      scrollToBottom();
      return { wrapper, bubble };
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

    function appendModelBadge(wrapper, modelName) {
      const isCloud = modelName.toLowerCase().includes('claude');
      const badge   = document.createElement('span');
      badge.className = 'model-badge ' + (isCloud ? 'cloud' : 'local');
      badge.textContent = modelName;
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
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    // ── Main send logic ─────────────────────────────────────────

    async function send() {
      const text = input.value.trim();
      if (!text) return;

      appendMessage('user', text);
      input.value = '';
      input.style.height = 'auto';
      setLocked(true);

      const { wrapper, bubble } = appendMessage('assistant');

      const cursor = document.createElement('span');
      cursor.className = 'cursor';
      bubble.appendChild(cursor);

      let accumulated = '';

      try {
        const resp = await fetch('/chat', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body:    JSON.stringify({ message: text }),
        });

        if (!resp.ok) throw new Error(`Server error: HTTP ${resp.status}`);

        const reader  = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer    = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop(); // hold onto any incomplete line

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;

            const raw = line.slice(6).trim();
            if (raw === '[DONE]') continue;

            let parsed;
            try { parsed = JSON.parse(raw); } catch { continue; }

            if (parsed.model) {
              appendModelBadge(wrapper, parsed.model);
            }

            if (parsed.text) {
              accumulated += parsed.text;
              bubble.innerHTML = marked.parse(accumulated);
              bubble.appendChild(cursor); // keep cursor at end
              scrollToBottom();
            }
          }
        }
      } catch (err) {
        bubble.textContent = `⚠ ${err.message}`;
      } finally {
        cursor.remove();
        setLocked(false);
        input.focus();
      }
    }

    async function startNewChat() {
      setLocked(true);
      try {
        const resp = await fetch('/chat/session', {
          method: 'DELETE',
          credentials: 'same-origin',
        });
        if (!resp.ok) throw new Error(`Server error: HTTP ${resp.status}`);
        resetMessagesView();
      } catch (err) {
        appendMessage('assistant', `⚠ Unable to reset chat session: ${err.message}`);
      } finally {
        setLocked(false);
        input.focus();
      }
    }