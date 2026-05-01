// persona.js — Phase 11: Persona configuration panel
// Loads current persona settings from GET /persona on page load and
// saves changes via POST /persona.

(function () {
  "use strict";

  const nameInput = document.getElementById("persona-name");
  const styleSelect = document.getElementById("persona-style");
  const warmthSlider = document.getElementById("persona-warmth");
  const warmthValue = document.getElementById("persona-warmth-value");
  const saveBtn = document.getElementById("persona-save-btn");
  const saveStatus = document.getElementById("persona-save-status");

  if (!nameInput || !styleSelect || !warmthSlider || !saveBtn) {
    // Panel not present in this build.
    return;
  }

  warmthSlider.addEventListener("input", () => {
    warmthValue.textContent = warmthSlider.value;
  });

  function _setStatus(msg, isError) {
    saveStatus.textContent = msg;
    saveStatus.style.color = isError ? "var(--danger, #e55)" : "var(--accent, #6cf)";
    setTimeout(() => {
      saveStatus.textContent = "";
    }, 3000);
  }

  function _applyConfig(config) {
    if (config.name !== undefined) nameInput.value = config.name || "";
    if (config.style !== undefined) styleSelect.value = config.style || "neutral";
    if (config.warmth !== undefined) {
      warmthSlider.value = config.warmth;
      warmthValue.textContent = config.warmth;
    }
  }

  async function loadPersona() {
    try {
      const resp = await fetch("/persona", { credentials: "include" });
      if (resp.ok) {
        const config = await resp.json();
        _applyConfig(config);
      }
    } catch (_err) {
      // Non-fatal: panel just shows defaults.
    }
  }

  saveBtn.addEventListener("click", async () => {
    const payload = {
      name: nameInput.value.trim(),
      style: styleSelect.value,
      warmth: parseInt(warmthSlider.value, 10),
    };
    try {
      const resp = await fetch("/persona", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (resp.ok) {
        const config = await resp.json();
        _applyConfig(config);
        _setStatus("Saved", false);
      } else {
        const err = await resp.json().catch(() => ({}));
        _setStatus(err.error || "Save failed", true);
      }
    } catch (exc) {
      _setStatus("Network error", true);
    }
  });

  // Load on page initialisation.
  loadPersona();
})();
