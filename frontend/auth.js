/**
 * auth.js — Client-side authentication helper.
 *
 * Flow:
 *   1. On page load, check localStorage for a token.
 *   2. Validate it against GET /auth/me.
 *   3. If valid → hide login overlay, show app.
 *   4. If invalid/missing → show login overlay.
 *
 * All API requests from message.js / voice.js call apiFetch() which
 * automatically attaches the Authorization header.  On a 401 response,
 * it clears the token and shows the login overlay.
 */

const AUTH_TOKEN_KEY = 'assistant_auth_token';
const AUTH_USER_KEY  = 'assistant_auth_user';

// ── Token storage ─────────────────────────────────────────────────────────────

export function getToken() {
  return localStorage.getItem(AUTH_TOKEN_KEY);
}

function setToken(token, user) {
  localStorage.setItem(AUTH_TOKEN_KEY, token);
  localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
}

function clearToken() {
  localStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_USER_KEY);
}

export function getStoredUser() {
  try {
    return JSON.parse(localStorage.getItem(AUTH_USER_KEY) || 'null');
  } catch {
    return null;
  }
}

// ── Authenticated fetch wrapper ───────────────────────────────────────────────

/**
 * Drop-in replacement for fetch() that attaches the bearer token and handles
 * 401 responses by clearing the session and showing the login overlay.
 */
export async function apiFetch(url, options = {}) {
  const token = getToken();
  const headers = { ...(options.headers || {}) };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const response = await fetch(url, { ...options, headers });

  if (response.status === 401) {
    clearToken();
    showLoginOverlay('Your session has expired. Please log in again.');
    throw new Error('Unauthorized');
  }

  return response;
}

// ── Auth API calls ────────────────────────────────────────────────────────────

async function register(username, password, persistent) {
  const res = await fetch('/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password, persistent }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Registration failed.');
  return data;
}

async function login(username, password, persistent) {
  const res = await fetch('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password, persistent }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Login failed.');
  return data;
}

async function verifyStoredToken() {
  const token = getToken();
  if (!token) return false;
  try {
    const res = await fetch('/auth/me', {
      headers: { 'Authorization': `Bearer ${token}` },
    });
    if (!res.ok) { clearToken(); return false; }
    const user = await res.json();
    localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
    return true;
  } catch {
    return false;
  }
}

export async function logout() {
  const token = getToken();
  if (token) {
    try {
      await fetch('/auth/logout', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
      });
    } catch { /* best-effort */ }
  }
  clearToken();
  showLoginOverlay();
}

// ── Login overlay UI ──────────────────────────────────────────────────────────

function showLoginOverlay(message = '') {
  const overlay = document.getElementById('login-overlay');
  const appShell = document.getElementById('app-shell');
  const errorEl = document.getElementById('login-error');

  if (overlay) overlay.style.display = 'flex';
  if (appShell) appShell.style.display = 'none';
  if (errorEl) errorEl.textContent = message;
}

function hideLoginOverlay() {
  const overlay = document.getElementById('login-overlay');
  const appShell = document.getElementById('app-shell');

  if (overlay) overlay.style.display = 'none';
  if (appShell) appShell.style.display = '';
}

function updateUserDisplay() {
  const user = getStoredUser();
  const el = document.getElementById('auth-username');
  if (el && user) el.textContent = user.username;
}

// ── Init ──────────────────────────────────────────────────────────────────────

export async function initAuth() {
  const valid = await verifyStoredToken();
  if (valid) {
    hideLoginOverlay();
    updateUserDisplay();
    return;
  }
  showLoginOverlay();
}

// ── Event wiring (runs after DOM is ready) ────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  // Tabs: login ↔ register
  document.getElementById('tab-login')?.addEventListener('click', () => {
    document.getElementById('tab-login').classList.add('active');
    document.getElementById('tab-register').classList.remove('active');
    document.getElementById('login-form').style.display = '';
    document.getElementById('register-form').style.display = 'none';
    document.getElementById('login-error').textContent = '';
  });
  document.getElementById('tab-register')?.addEventListener('click', () => {
    document.getElementById('tab-register').classList.add('active');
    document.getElementById('tab-login').classList.remove('active');
    document.getElementById('register-form').style.display = '';
    document.getElementById('login-form').style.display = 'none';
    document.getElementById('login-error').textContent = '';
  });

  // Login form
  document.getElementById('login-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = document.getElementById('login-username').value.trim();
    const password = document.getElementById('login-password').value;
    const persistent = document.getElementById('login-persistent')?.checked ?? false;
    const errorEl = document.getElementById('login-error');
    errorEl.textContent = '';
    try {
      const data = await login(username, password, persistent);
      setToken(data.token, { user_id: data.user_id, username: data.username });
      hideLoginOverlay();
      updateUserDisplay();
      // Reload so message.js reinitialises with a fresh session for this user.
      window.location.reload();
    } catch (err) {
      errorEl.textContent = err.message;
    }
  });

  // Register form
  document.getElementById('register-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = document.getElementById('reg-username').value.trim();
    const password = document.getElementById('reg-password').value;
    const persistent = document.getElementById('reg-persistent')?.checked ?? false;
    const errorEl = document.getElementById('login-error');
    errorEl.textContent = '';
    try {
      const data = await register(username, password, persistent);
      setToken(data.token, { user_id: data.user_id, username: data.username });
      hideLoginOverlay();
      updateUserDisplay();
      window.location.reload();
    } catch (err) {
      errorEl.textContent = err.message;
    }
  });

  // Logout button
  document.getElementById('logout-btn')?.addEventListener('click', async () => {
    await logout();
  });

  // Kick off auth check
  initAuth();

  // Expose apiFetch globally so non-module scripts (message.js, voice.js) can use it.
  window.apiFetch = apiFetch;
});
