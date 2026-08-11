/* Offline interface preview.
 *
 * This build deliberately makes no network request. It exists so palette,
 * hierarchy, type, mobile layout and source cards can be judged without
 * paying for a model response. Restore the SSE transport only after the visual
 * direction is approved.
 */

const LAYER_NAME = {
  manufacturer_datasheet: "manufacturer",
  materials_science: "materials science",
  conservation_literature: "conservation",
  collection_precedent: "collection",
};

const ATTRIBUTION = [
  [/manufactur|data ?sheet|smooth-?on|the maker|vendor/i, "manufacturer_datasheet"],
  [/conservation|conservator|restorer/i, "conservation_literature"],
  [/collection|tate|precedent|holdings/i, "collection_precedent"],
  [/study|studies|research|peer-reviewed|paper|literature|experiment/i, "materials_science"],
];

const PREVIEW_ANSWER = `**Interface preview.** No retrieval or Claude request was made. This local sample lets you judge the pixel typography, spacing, palette and reading experience without spending anything.

In the connected version, the answer will stream into this solid panel. Manufacturer, materials-science, conservation and collection evidence will remain visibly separate, with the retrieved source chunks below.`;

const PREVIEW_HITS = [
  {
    source_type: "manufacturer_datasheet",
    evidence_kind: "accelerated testing",
    title: "Manufacturer source card — interface placeholder",
    text: "Sample passage area. The connected interface will place the retrieved text here without changing its wording.",
    chunk_id: "preview/manufacturer/001",
  },
  {
    source_type: "materials_science",
    evidence_kind: "controlled specimens",
    title: "Materials-science source card — interface placeholder",
    text: "Sample passage area for checking hierarchy, expansion behaviour and body-text contrast.",
    chunk_id: "preview/materials-science/002",
  },
  {
    source_type: "conservation_literature",
    evidence_kind: "examined objects",
    title: "Conservation source card — interface placeholder",
    text: "This is local demonstration copy only. It is not presented as evidence about a material.",
    chunk_id: "preview/conservation/003",
  },
  {
    source_type: "collection_precedent",
    evidence_kind: "held in a collection",
    title: "Collection precedent card — interface placeholder",
    text: "The final card will carry the exact source chunk id used to build the answer.",
    chunk_id: "preview/collection/004",
  },
];

const form = document.getElementById("ask");
const input = document.getElementById("q");
const go = document.getElementById("go");
const rewriteEl = document.getElementById("rewrite");
const resultWindow = document.getElementById("result-window");
const answerEl = document.getElementById("answer");
const metaEl = document.getElementById("meta");
const errorEl = document.getElementById("error");
const specimensEl = document.getElementById("specimens");
const listEl = document.getElementById("specimen-list");
const themeButtons = document.querySelectorAll("[data-theme-choice]");
const apiSettingsForm = document.getElementById("api-settings");
const apiProvider = document.getElementById("api-provider");
const providerPickerButton = document.getElementById("provider-picker-button");
const providerPickerLabel = document.getElementById("provider-picker-label");
const providerPickerList = document.getElementById("provider-picker-list");
const apiModel = document.getElementById("api-model");
const apiKey = document.getElementById("api-key");
const apiBaseUrl = document.getElementById("api-base-url");
const apiProtocol = document.getElementById("api-protocol");
const apiKeyStatus = document.getElementById("api-key-status");
const toggleApiKey = document.getElementById("toggle-api-key");
const forgetApiKey = document.getElementById("forget-api-key");
const historyList = document.getElementById("history-list");
const historyEmpty = document.getElementById("history-empty");
const clearHistory = document.getElementById("clear-history");
const desktopWindows = document.querySelectorAll(".draggable-window");
const pixelCaret = document.getElementById("pixel-caret");
const windowToggleButtons = document.querySelectorAll("[data-window-toggle]");
const windowOpenButtons = document.querySelectorAll("[data-window-open]");
const startButton = document.getElementById("start-button");
const startMenu = document.getElementById("start-menu");
const settingsMenuTrigger = document.getElementById("settings-menu-trigger");
const desktopSettingsButton = document.getElementById("desktop-settings");
const windowStates = new Map();
const COMPACT_BREAKPOINT = 600;
let normalStackOrder = 100;

function setTheme(theme) {
  const selected = theme === "neutral" ? "neutral" : "contrast";
  document.body.dataset.theme = selected;
  for (const button of themeButtons) {
    button.setAttribute("aria-pressed", String(button.dataset.themeChoice === selected));
  }
  try {
    localStorage.setItem("artmat-theme", selected);
  } catch (_) {
    // A blocked storage API should not block a visual preference.
  }
}

for (const button of themeButtons) {
  button.addEventListener("click", () => {
    setTheme(button.dataset.themeChoice);
    if (button.closest("#start-menu")) {
      startMenu.hidden = true;
      startButton.setAttribute("aria-expanded", "false");
    }
  });
}

try {
  setTheme(localStorage.getItem("artmat-theme") || "contrast");
} catch (_) {
  setTheme("contrast");
}

const PROVIDER_PROTOCOL = {
  openai: "OPENAI RESPONSES / CHAT",
  anthropic: "ANTHROPIC MESSAGES",
  google: "GOOGLE GENERATIVE LANGUAGE",
  mistral: "OPENAI-COMPATIBLE",
  xai: "OPENAI-COMPATIBLE",
  deepseek: "OPENAI-COMPATIBLE",
  openrouter: "OPENAI-COMPATIBLE ROUTER",
  groq: "OPENAI-COMPATIBLE",
  cohere: "COHERE CHAT",
  together: "OPENAI-COMPATIBLE",
  perplexity: "OPENAI-COMPATIBLE",
  "openai-compatible": "CUSTOM OPENAI-COMPATIBLE",
  "anthropic-compatible": "CUSTOM ANTHROPIC-COMPATIBLE",
};

let runtimeApiKey = "";
let questionHistory = [];
let historyClearArmed = false;
let historyClearTimer = 0;

function updateProviderProtocol() {
  apiProtocol.textContent = `PROTOCOL: ${PROVIDER_PROTOCOL[apiProvider.value] || "CUSTOM"}`;
}

function saveConnectionPreferences() {
  try {
    localStorage.setItem("artmat-ai-connection", JSON.stringify({
      provider: apiProvider.value,
      model: apiModel.value.trim(),
      baseUrl: apiBaseUrl.value.trim(),
    }));
  } catch (_) {
    // Provider preferences are optional; the secret is never stored here.
  }
}

function restoreConnectionPreferences() {
  try {
    const saved = JSON.parse(localStorage.getItem("artmat-ai-connection") || "null");
    if (!saved) return;
    if ([...apiProvider.options].some((option) => option.value === saved.provider)) {
      apiProvider.value = saved.provider;
    }
    apiModel.value = saved.model || "";
    apiBaseUrl.value = saved.baseUrl || "";
  } catch (_) {
    // Ignore malformed or unavailable local preferences.
  }
}

function syncProviderPicker() {
  const selectedOption = apiProvider.options[apiProvider.selectedIndex];
  providerPickerLabel.textContent = selectedOption?.text || "Select provider";
  for (const option of providerPickerList.querySelectorAll(".provider-option")) {
    option.setAttribute("aria-selected", String(option.dataset.value === apiProvider.value));
  }
}

function positionProviderList() {
  const rect = providerPickerButton.getBoundingClientRect();
  const availableBelow = window.innerHeight - rect.bottom - 8;
  const preferredHeight = Math.min(368, window.innerHeight - 16);
  const opensUp = availableBelow < Math.min(260, preferredHeight) && rect.top > availableBelow;
  providerPickerList.style.left = `${Math.max(8, Math.min(rect.left, window.innerWidth - rect.width - 8))}px`;
  providerPickerList.style.width = `${rect.width}px`;
  providerPickerList.style.maxHeight = `${opensUp ? Math.max(120, rect.top - 8) : Math.max(120, availableBelow)}px`;
  providerPickerList.style.top = opensUp ? "auto" : `${rect.bottom + 2}px`;
  providerPickerList.style.bottom = opensUp ? `${window.innerHeight - rect.top + 2}px` : "auto";
}

function closeProviderPicker() {
  providerPickerList.hidden = true;
  providerPickerButton.setAttribute("aria-expanded", "false");
}

for (const sourceOption of apiProvider.options) {
  const option = document.createElement("button");
  option.type = "button";
  option.className = "provider-option";
  option.setAttribute("role", "option");
  option.dataset.value = sourceOption.value;
  option.textContent = sourceOption.text;
  option.addEventListener("click", () => {
    apiProvider.value = sourceOption.value;
    apiProvider.dispatchEvent(new Event("change", { bubbles: true }));
    closeProviderPicker();
    providerPickerButton.focus();
  });
  providerPickerList.append(option);
}

providerPickerButton.addEventListener("click", () => {
  const opening = providerPickerList.hidden;
  if (!opening) {
    closeProviderPicker();
    return;
  }
  providerPickerList.hidden = false;
  providerPickerButton.setAttribute("aria-expanded", "true");
  positionProviderList();
  providerPickerList.querySelector('[aria-selected="true"]')?.focus();
});

providerPickerList.addEventListener("keydown", (event) => {
  const options = [...providerPickerList.querySelectorAll(".provider-option")];
  const index = options.indexOf(document.activeElement);
  if (event.key === "Escape") {
    closeProviderPicker();
    providerPickerButton.focus();
  } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    const direction = event.key === "ArrowDown" ? 1 : -1;
    options[(index + direction + options.length) % options.length]?.focus();
  }
});

document.addEventListener("pointerdown", (event) => {
  if (!providerPickerButton.contains(event.target) && !providerPickerList.contains(event.target)) {
    closeProviderPicker();
  }
});
window.addEventListener("resize", closeProviderPicker);

restoreConnectionPreferences();
updateProviderProtocol();
syncProviderPicker();

apiProvider.addEventListener("change", () => {
  updateProviderProtocol();
  saveConnectionPreferences();
  syncProviderPicker();
});
for (const field of [apiModel, apiBaseUrl]) {
  field.addEventListener("change", saveConnectionPreferences);
}

apiSettingsForm.addEventListener("submit", (event) => {
  event.preventDefault();
  runtimeApiKey = apiKey.value.trim();
  saveConnectionPreferences();
  apiKeyStatus.textContent = runtimeApiKey
    ? `KEY STATUS: LOADED IN MEMORY / ${apiProvider.options[apiProvider.selectedIndex].text}`
    : "KEY STATUS: ENTER A KEY FIRST";
});

toggleApiKey.addEventListener("click", () => {
  const showing = apiKey.type === "text";
  apiKey.type = showing ? "password" : "text";
  toggleApiKey.textContent = showing ? "SHOW" : "HIDE";
  toggleApiKey.setAttribute("aria-pressed", String(!showing));
});

forgetApiKey.addEventListener("click", () => {
  runtimeApiKey = "";
  apiKey.value = "";
  apiKey.type = "password";
  toggleApiKey.textContent = "SHOW";
  toggleApiKey.setAttribute("aria-pressed", "false");
  apiKeyStatus.textContent = "KEY STATUS: NOT LOADED";
});

function persistHistory() {
  try {
    localStorage.setItem("artmat-question-history", JSON.stringify(questionHistory));
  } catch (_) {
    // History is a local convenience and may be unavailable in private mode.
  }
}

function renderHistory() {
  historyList.replaceChildren();
  historyEmpty.hidden = questionHistory.length > 0;
  for (const entry of questionHistory) {
    const item = document.createElement("li");
    item.className = "history-item";

    const reuse = document.createElement("button");
    reuse.type = "button";
    reuse.className = "history-reuse";
    const question = document.createElement("span");
    question.className = "history-question";
    question.textContent = entry.question;
    const time = document.createElement("time");
    time.className = "history-time";
    time.dateTime = entry.createdAt;
    time.textContent = new Date(entry.createdAt).toLocaleString([], { dateStyle: "short", timeStyle: "short" });
    reuse.append(question, time);
    reuse.addEventListener("click", () => {
      input.value = entry.question;
      const askWindow = windowById("ask");
      setMinimized(askWindow, false);
      bringToFront(askWindow);
      input.focus();
      updatePixelCaret();
    });

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "history-delete";
    remove.setAttribute("aria-label", `Delete history entry: ${entry.question}`);
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      questionHistory = questionHistory.filter((candidate) => candidate.id !== entry.id);
      persistHistory();
      renderHistory();
    });
    item.append(reuse, remove);
    historyList.appendChild(item);
  }
}

function restoreHistory() {
  try {
    const saved = JSON.parse(localStorage.getItem("artmat-question-history") || "[]");
    questionHistory = Array.isArray(saved) ? saved.slice(0, 30) : [];
  } catch (_) {
    questionHistory = [];
  }
  renderHistory();
}

function recordQuestion(question) {
  const duplicateIndex = questionHistory.findIndex((entry) => entry.question === question);
  if (duplicateIndex >= 0) questionHistory.splice(duplicateIndex, 1);
  questionHistory.unshift({
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    question,
    createdAt: new Date().toISOString(),
  });
  questionHistory = questionHistory.slice(0, 30);
  persistHistory();
  renderHistory();
}

clearHistory.addEventListener("click", () => {
  if (!historyClearArmed) {
    historyClearArmed = true;
    clearHistory.textContent = "CONFIRM CLEAR";
    window.clearTimeout(historyClearTimer);
    historyClearTimer = window.setTimeout(() => {
      historyClearArmed = false;
      clearHistory.textContent = "CLEAR ALL";
    }, 3000);
    return;
  }
  window.clearTimeout(historyClearTimer);
  historyClearArmed = false;
  questionHistory = [];
  persistHistory();
  renderHistory();
  clearHistory.textContent = "CLEAR ALL";
});

restoreHistory();

function stateFor(windowElement) {
  return windowStates.get(windowElement);
}

function windowById(id) {
  return document.querySelector(`.draggable-window[data-window="${id}"]`);
}

function ensureWindowGeometry(windowElement) {
  const state = stateFor(windowElement);
  if (!state || state.initialized || window.innerWidth <= COMPACT_BREAKPOINT || windowElement.hidden) return;

  const wasMinimized = windowElement.classList.contains("is-minimized");
  if (wasMinimized) windowElement.classList.remove("is-minimized");
  const rect = windowElement.getBoundingClientRect();
  if (wasMinimized) windowElement.classList.add("is-minimized");
  if (!rect.width || !rect.height) return;

  const minWidth = Number(windowElement.dataset.minWidth || 180);
  const minHeight = Number(windowElement.dataset.minHeight || 70);
  const maxWidth = Number(windowElement.dataset.maxWidth || window.innerWidth - 16);
  const maxHeight = Number(windowElement.dataset.maxHeight || window.innerHeight - 62);
  const initialWidth = Math.max(minWidth, Math.min(rect.width, maxWidth, window.innerWidth - 16));
  const initialHeight = Math.max(minHeight, Math.min(rect.height, maxHeight, window.innerHeight - 62));

  state.x = rect.left;
  state.y = rect.top;
  state.initialized = true;
  windowElement.style.position = "fixed";
  windowElement.style.left = `${rect.left}px`;
  windowElement.style.top = `${rect.top}px`;
  windowElement.style.right = "auto";
  windowElement.style.bottom = "auto";
  windowElement.style.width = `${initialWidth}px`;
  windowElement.style.height = `${initialHeight}px`;
  windowElement.style.margin = "0";
  windowElement.style.transform = "none";
}

function applyPosition(windowElement) {
  const state = stateFor(windowElement);
  if (window.innerWidth <= COMPACT_BREAKPOINT) return;
  ensureWindowGeometry(windowElement);
  windowElement.style.left = `${state.x}px`;
  windowElement.style.top = `${state.y}px`;
  windowElement.style.transform = "none";
}

function windowIsOpen(windowElement) {
  if (!windowElement || windowElement.hidden || stateFor(windowElement)?.minimized) return false;
  let ancestor = windowElement.parentElement?.closest(".draggable-window");
  while (ancestor) {
    if (ancestor.hidden || stateFor(ancestor)?.minimized) return false;
    ancestor = ancestor.parentElement?.closest(".draggable-window");
  }
  return true;
}

function updateTaskButtons() {
  for (const button of windowToggleButtons) {
    const windowElement = windowById(button.dataset.windowToggle);
    if (!windowElement) continue;
    const visible = windowIsOpen(windowElement);
    button.hidden = !visible;
    button.disabled = false;
    button.setAttribute("aria-pressed", String(visible && windowElement.classList.contains("is-active")));
  }
}

function bringToFront(windowElement) {
  if (!windowElement || stateFor(windowElement)?.minimized) return;
  for (const candidate of desktopWindows) candidate.classList.remove("is-active");

  normalStackOrder += 1;
  const askWindow = windowById("ask");
  for (const candidate of desktopWindows) {
    if (candidate === askWindow || candidate === windowElement) continue;
    const state = stateFor(candidate);
    candidate.style.zIndex = String(Math.min(899, state?.order || 100));
  }

  if (windowElement === askWindow) {
    askWindow.style.zIndex = "1001";
  } else {
    if (askWindow && !stateFor(askWindow)?.minimized) askWindow.style.zIndex = "1000";
    windowElement.style.zIndex = "1001";
  }

  const state = stateFor(windowElement);
  if (state) state.order = normalStackOrder;
  windowElement.classList.add("is-active");
  updateTaskButtons();
}

function setMinimized(windowElement, minimized) {
  const state = stateFor(windowElement);
  if (!state) return;
  if (!minimized) {
    const ancestor = windowElement.parentElement?.closest(".draggable-window");
    if (ancestor && !windowIsOpen(ancestor)) setMinimized(ancestor, false);
  }
  state.minimized = minimized;
  windowElement.classList.toggle("is-minimized", minimized);
  if (!minimized) {
    ensureWindowGeometry(windowElement);
    applyPosition(windowElement);
    bringToFront(windowElement);
  } else {
    windowElement.classList.remove("is-active");
    const askWindow = windowById("ask");
    if (askWindow && askWindow !== windowElement && !stateFor(askWindow)?.minimized) bringToFront(askWindow);
  }
  updateTaskButtons();
}

function positionWindow(windowElement, proposedX, proposedY) {
  const state = stateFor(windowElement);
  const handle = windowElement.querySelector(".drag-handle");
  ensureWindowGeometry(windowElement);
  state.x = proposedX;
  state.y = proposedY;
  applyPosition(windowElement);

  const rect = handle.getBoundingClientRect();
  const margin = 8;
  const taskbarTop = window.innerHeight - 48;
  if (rect.left < margin) state.x += margin - rect.left;
  if (rect.right > window.innerWidth - margin) state.x -= rect.right - (window.innerWidth - margin);
  if (rect.top < margin) state.y += margin - rect.top;
  if (rect.bottom > taskbarTop) state.y -= rect.bottom - taskbarTop;
  applyPosition(windowElement);
}

function toggleMaximize(windowElement) {
  const state = stateFor(windowElement);
  ensureWindowGeometry(windowElement);
  if (!state.maximized) {
    state.restore = {
      x: state.x,
      y: state.y,
      width: windowElement.style.width,
      height: windowElement.style.height,
    };
    const maximizedWidth = Math.min(
      window.innerWidth - 20,
      Number(windowElement.dataset.maxWidth || window.innerWidth - 20),
    );
    const maximizedHeight = Math.min(
      window.innerHeight - 62,
      Number(windowElement.dataset.maxHeight || window.innerHeight - 62),
    );
    windowElement.style.width = `${maximizedWidth}px`;
    windowElement.style.height = `${maximizedHeight}px`;
    applyPosition(windowElement);
    const maximizedRect = windowElement.getBoundingClientRect();
    state.x += 10 - maximizedRect.left;
    state.y += 10 - maximizedRect.top;
    state.maximized = true;
    windowElement.classList.add("is-maximized");
  } else {
    state.x = state.restore.x;
    state.y = state.restore.y;
    windowElement.style.width = state.restore.width;
    windowElement.style.height = state.restore.height;
    state.maximized = false;
    windowElement.classList.remove("is-maximized");
  }
  applyPosition(windowElement);
  bringToFront(windowElement);
}

function ensureWindowControls(windowElement, handle) {
  let controls = handle.querySelector(".dialog-buttons");
  if (!controls) {
    controls = document.createElement("span");
    controls.className = "dialog-buttons";
    handle.appendChild(controls);
  }
  const actions = [
    ["minimize", "_", "Minimize"],
    ["maximize", "□", "Maximize"],
    ["close", "×", "Close"],
  ];
  for (const [action, glyph, label] of actions) {
    if (controls.querySelector(`[data-window-action="${action}"]`)) continue;
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.windowAction = action;
    button.setAttribute("aria-label", `${label} ${windowElement.dataset.windowLabel || "window"}`);
    button.textContent = glyph;
    controls.appendChild(button);
  }
}

function addResizeHandles(windowElement) {
  for (const edge of ["n", "e", "s", "w", "nw", "ne", "se", "sw"]) {
    const resizeHandle = document.createElement("span");
    resizeHandle.className = "resize-handle";
    resizeHandle.dataset.edge = edge;
    resizeHandle.setAttribute("aria-hidden", "true");
    windowElement.appendChild(resizeHandle);

    resizeHandle.addEventListener("mousedown", (event) => {
      if (window.innerWidth <= COMPACT_BREAKPOINT) return;
      event.preventDefault();
      event.stopPropagation();
      ensureWindowGeometry(windowElement);
      bringToFront(windowElement);
      windowElement.classList.add("is-resizing");

      const state = stateFor(windowElement);
      const rect = windowElement.getBoundingClientRect();
      const start = {
        pointerX: event.clientX,
        pointerY: event.clientY,
        x: state.x,
        y: state.y,
        width: rect.width,
        height: rect.height,
        left: rect.left,
        top: rect.top,
        right: rect.right,
        bottom: rect.bottom,
      };
      const minWidth = Number(windowElement.dataset.minWidth || (windowElement.dataset.window === "ask" ? 300 : 180));
      const minHeight = Number(windowElement.dataset.minHeight || 70);
      const maxWidth = Number(windowElement.dataset.maxWidth || window.innerWidth - 16);
      const maxHeight = Number(windowElement.dataset.maxHeight || window.innerHeight - 62);

      const move = (moveEvent) => {
        const dx = moveEvent.clientX - start.pointerX;
        const dy = moveEvent.clientY - start.pointerY;
        let nextWidth = start.width;
        let nextHeight = start.height;

        if (edge.includes("e")) nextWidth = Math.max(minWidth, start.width + dx);
        if (edge.includes("s")) nextHeight = Math.max(minHeight, start.height + dy);
        if (edge.includes("w")) {
          nextWidth = Math.max(minWidth, start.width - dx);
        }
        if (edge.includes("n")) {
          nextHeight = Math.max(minHeight, start.height - dy);
        }

        nextWidth = Math.min(nextWidth, maxWidth, window.innerWidth - 16);
        nextHeight = Math.min(nextHeight, maxHeight, window.innerHeight - 62);
        state.x = start.x;
        state.y = start.y;
        applyPosition(windowElement);
        windowElement.style.width = `${nextWidth}px`;
        windowElement.style.height = `${nextHeight}px`;

        // A window may live in a bottom/right anchored layout container. After
        // changing its dimensions, compensate for that container's reflow so
        // the two edges the user did not grab remain visually fixed.
        const resizedRect = windowElement.getBoundingClientRect();
        const desiredLeft = edge.includes("w") ? start.right - nextWidth : start.left;
        const desiredTop = edge.includes("n") ? start.bottom - nextHeight : start.top;
        positionWindow(
          windowElement,
          start.x + desiredLeft - resizedRect.left,
          start.y + desiredTop - resizedRect.top,
        );
      };

      const stop = () => {
        windowElement.classList.remove("is-resizing");
        document.removeEventListener("mousemove", move);
        document.removeEventListener("mouseup", stop);
      };
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", stop);
    });
  }
}

for (const windowElement of desktopWindows) {
  const handle = windowElement.querySelector(".drag-handle");
  windowStates.set(windowElement, {
    x: 0,
    y: 0,
    initialized: false,
    order: normalStackOrder += 1,
    minimized: windowElement.classList.contains("is-minimized"),
    maximized: false,
    restore: null,
  });
  ensureWindowGeometry(windowElement);
  ensureWindowControls(windowElement, handle);
  addResizeHandles(windowElement);

  windowElement.addEventListener("mousedown", (event) => {
    if (event.target.closest(".draggable-window") === windowElement) bringToFront(windowElement);
  });
  handle.addEventListener("mousedown", (event) => {
    if (event.target.closest("button") || window.innerWidth <= COMPACT_BREAKPOINT) return;
    event.preventDefault();
    bringToFront(windowElement);
    windowElement.classList.add("is-dragging");
    const state = stateFor(windowElement);
    const startPointerX = event.clientX;
    const startPointerY = event.clientY;
    const startWindowX = state.x;
    const startWindowY = state.y;
    const move = (moveEvent) => positionWindow(
      windowElement,
      startWindowX + moveEvent.clientX - startPointerX,
      startWindowY + moveEvent.clientY - startPointerY,
    );
    const stop = () => {
      windowElement.classList.remove("is-dragging");
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", stop);
    };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", stop);
  });

  handle.addEventListener("dblclick", (event) => {
    if (!event.target.closest("button")) toggleMaximize(windowElement);
  });

  handle.addEventListener("keydown", (event) => {
    const movement = {
      ArrowLeft: [-16, 0],
      ArrowRight: [16, 0],
      ArrowUp: [0, -16],
      ArrowDown: [0, 16],
    }[event.key];
    if (!movement || window.innerWidth <= COMPACT_BREAKPOINT) return;
    event.preventDefault();
    bringToFront(windowElement);
    const state = stateFor(windowElement);
    positionWindow(windowElement, state.x + movement[0], state.y + movement[1]);
  });

  handle.querySelectorAll("[data-window-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.dataset.windowAction;
      if (action === "minimize" || action === "close") setMinimized(windowElement, true);
      if (action === "maximize") toggleMaximize(windowElement);
    });
  });
}

function applyInitialDesktopCascade() {
  if (window.innerWidth <= COMPACT_BREAKPOINT) return;

  // Keep the left desktop rail visible, then offset each open window like a
  // stack of Win95 error dialogs. Sizes stay inside the current viewport so
  // the composition is intentional rather than overflow-driven.
  const startX = 108;
  const baseWidth = Math.min(560, window.innerWidth - startX - 16);
  const availableHeight = window.innerHeight - 82;
  const stackX = Math.max(118, window.innerWidth - 497);
  const stackY = Math.max(190, Math.min(300, window.innerHeight - 330));
  const cascade = [
    ["artmat", 0, 18, baseWidth, Math.min(500, availableHeight), 110],
    ["examples", stackX - startX, stackY, Math.min(460, baseWidth - 34), Math.min(260, availableHeight - 70), 120],
    ["palette", stackX - startX + 14, stackY + 22, Math.min(340, baseWidth - 72), 200, 130],
    ["font", stackX - startX + 28, stackY + 44, Math.min(360, baseWidth - 86), 210, 140],
    ["api", stackX - startX + 42, stackY + 66, Math.min(380, baseWidth - 100), 210, 150],
    ["history", stackX - startX + 56, stackY + 88, Math.min(360, baseWidth - 114), 180, 160],
    ["ask", stackX - startX + 70, stackY + 110, 420, 124, 170],
  ];

  for (const [id, offsetX, y, width, height, order] of cascade) {
    const windowElement = windowById(id);
    const state = stateFor(windowElement);
    if (!windowElement || !state) continue;

    state.minimized = false;
    state.order = order;
    windowElement.classList.remove("is-minimized");
    windowElement.style.width = `${width}px`;
    windowElement.style.height = `${Math.max(108, height)}px`;
    state.x = startX + offsetX;
    state.y = y;
    applyPosition(windowElement);
    positionWindow(windowElement, state.x, state.y);
  }
}

applyInitialDesktopCascade();

for (const button of windowToggleButtons) {
  button.addEventListener("click", () => {
    const windowElement = windowById(button.dataset.windowToggle);
    if (!windowElement) return;
    if (stateFor(windowElement)?.minimized) {
      setMinimized(windowElement, false);
    } else {
      bringToFront(windowElement);
    }
  });
}

for (const button of windowOpenButtons) {
  button.addEventListener("click", () => {
    const windowElement = windowById(button.dataset.windowOpen);
    if (!windowElement) return;
    setMinimized(windowElement, false);
    startMenu.hidden = true;
    startButton.setAttribute("aria-expanded", "false");
  });
}

startButton.addEventListener("click", () => {
  startMenu.hidden = !startMenu.hidden;
  startButton.setAttribute("aria-expanded", String(!startMenu.hidden));
});

desktopSettingsButton.addEventListener("click", () => {
  startMenu.hidden = false;
  startButton.setAttribute("aria-expanded", "true");
  settingsMenuTrigger.focus();
});

for (const menuItem of document.querySelectorAll(".start-menu-item")) {
  const trigger = menuItem.querySelector(":scope > .start-submenu-trigger");
  if (!trigger) continue;
  menuItem.addEventListener("mouseenter", () => trigger.setAttribute("aria-expanded", "true"));
  menuItem.addEventListener("mouseleave", () => trigger.setAttribute("aria-expanded", "false"));
  menuItem.addEventListener("focusin", () => trigger.setAttribute("aria-expanded", "true"));
  menuItem.addEventListener("focusout", (event) => {
    if (!menuItem.contains(event.relatedTarget)) trigger.setAttribute("aria-expanded", "false");
  });
}

document.addEventListener("mousedown", (event) => {
  if (startMenu.hidden || startMenu.contains(event.target) || event.target === startButton) return;
  startMenu.hidden = true;
  startButton.setAttribute("aria-expanded", "false");
});

bringToFront(windowById("ask"));
updateTaskButtons();

document.querySelectorAll("#examples button").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.textContent.replace(/^\s*\d+\s*/, "").trim();
    input.focus();
    updatePixelCaret();
  });
});

const caretCanvas = document.createElement("canvas");
const caretContext = caretCanvas.getContext("2d");
const settingsCaretMap = new Map();

for (const field of apiSettingsForm.querySelectorAll("input")) {
  const caret = document.createElement("span");
  caret.className = "settings-pixel-caret";
  caret.setAttribute("aria-hidden", "true");
  caret.hidden = true;
  document.body.append(caret);
  settingsCaretMap.set(field, caret);
}

function updateSettingsCaret(field) {
  const caret = settingsCaretMap.get(field);
  if (!caret || document.activeElement !== field || !caretContext) {
    if (caret) caret.hidden = true;
    return;
  }

  const style = getComputedStyle(field);
  const caretIndex = field.selectionStart ?? field.value.length;
  const rawBeforeCaret = field.value.slice(0, caretIndex);
  const beforeCaret = field.type === "password" ? "•".repeat(rawBeforeCaret.length) : rawBeforeCaret;
  caretContext.font = style.font;
  const letterSpacing = Number.parseFloat(style.letterSpacing) || 0;
  const textWidth = caretContext.measureText(beforeCaret).width +
    Math.max(0, beforeCaret.length - 1) * letterSpacing;
  const paddingLeft = Number.parseFloat(style.paddingLeft) || 0;
  const rect = field.getBoundingClientRect();
  const clippingContainer = field.closest(".window-content");
  const clippingRect = clippingContainer?.getBoundingClientRect();
  const outsideViewport = rect.bottom <= 0 || rect.top >= window.innerHeight || rect.right <= 0 || rect.left >= window.innerWidth;
  const clippedByWindow = clippingRect && (
    rect.top < clippingRect.top ||
    rect.bottom > clippingRect.bottom ||
    rect.left < clippingRect.left ||
    rect.right > clippingRect.right
  );
  if (outsideViewport || clippedByWindow) {
    caret.hidden = true;
    return;
  }
  const caretWidth = 6;
  const maximumLeft = rect.right - caretWidth - 6;
  const left = Math.max(rect.left + paddingLeft, Math.min(maximumLeft, rect.left + paddingLeft + textWidth - field.scrollLeft));

  caret.style.left = `${left}px`;
  caret.style.top = `${rect.top + (rect.height - 20) / 2}px`;
  caret.hidden = false;
}

for (const field of settingsCaretMap.keys()) {
  for (const eventName of ["focus", "input", "keyup", "click", "scroll"]) {
    field.addEventListener(eventName, () => requestAnimationFrame(() => updateSettingsCaret(field)));
  }
  field.addEventListener("blur", () => { settingsCaretMap.get(field).hidden = true; });
}

function updateActiveSettingsCaret() {
  if (settingsCaretMap.has(document.activeElement)) updateSettingsCaret(document.activeElement);
}

window.addEventListener("resize", updateActiveSettingsCaret, { passive: true });
document.addEventListener("scroll", () => requestAnimationFrame(updateActiveSettingsCaret), {
  capture: true,
  passive: true,
});

function updatePixelCaret() {
  if (document.activeElement !== input || !caretContext) {
    pixelCaret.hidden = true;
    return;
  }

  const style = getComputedStyle(input);
  const caretIndex = input.selectionStart ?? input.value.length;
  const beforeCaret = input.value.slice(0, caretIndex);
  caretContext.font = style.font;
  const letterSpacing = Number.parseFloat(style.letterSpacing) || 0;
  const textWidth = caretContext.measureText(beforeCaret).width +
    Math.max(0, beforeCaret.length - 1) * letterSpacing;
  const paddingLeft = Number.parseFloat(style.paddingLeft) || 0;
  const maximumLeft = input.clientWidth - pixelCaret.offsetWidth - 6;
  const left = Math.max(paddingLeft, Math.min(maximumLeft, paddingLeft + textWidth - input.scrollLeft));

  pixelCaret.style.left = `${left}px`;
  pixelCaret.hidden = false;
}

for (const eventName of ["focus", "input", "keyup", "click", "scroll"]) {
  input.addEventListener(eventName, () => requestAnimationFrame(updatePixelCaret));
}
input.addEventListener("blur", () => { pixelCaret.hidden = true; });
document.addEventListener("selectionchange", () => {
  if (document.activeElement === input) requestAnimationFrame(updatePixelCaret);
  if (settingsCaretMap.has(document.activeElement)) {
    requestAnimationFrame(() => updateSettingsCaret(document.activeElement));
  }
});
function syncResponsiveGeometry() {
  if (window.innerWidth <= COMPACT_BREAKPOINT) {
    for (const windowElement of desktopWindows) {
      const state = stateFor(windowElement);
      state.initialized = false;
      for (const property of ["position", "left", "top", "right", "bottom", "width", "height", "margin", "transform"]) {
        windowElement.style.removeProperty(property);
      }
    }
  } else {
    for (const windowElement of desktopWindows) {
      ensureWindowGeometry(windowElement);
      if (!stateFor(windowElement).minimized && !windowElement.hidden) {
        positionWindow(windowElement, stateFor(windowElement).x, stateFor(windowElement).y);
      }
    }
  }
  updatePixelCaret();
}
window.addEventListener("resize", () => requestAnimationFrame(syncResponsiveGeometry));

function esc(value) {
  return String(value).replace(/[&<>"']/g, (character) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]));
}

function whoSpeaks(sentence) {
  for (const [pattern, layer] of ATTRIBUTION) {
    if (pattern.test(sentence)) return layer;
  }
  return null;
}

function renderAnswer(text, streaming = false) {
  const html = text.split(/\n{2,}/).map((paragraph) => {
    // Markdown closers can sit between punctuation and whitespace (`say.**
    // Two`). Keeping them in the boundary prevents adjacent sources from being
    // merged into one provenance tint.
    const inked = paragraph.split(/(?<=[.!?][*"')\]]{0,2})\s+/).map((sentence) => {
      const layer = whoSpeaks(sentence);
      const safe = esc(sentence).replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      return layer ? `<span class="src-${layer}">${safe}</span>` : safe;
    }).join(" ");
    return `<p>${inked.replace(/\n/g, "<br>")}</p>`;
  }).join("");
  answerEl.innerHTML = streaming ? `${html}<span class="cursor"></span>` : html;
}

function renderSpecimens(hits) {
  listEl.innerHTML = "";
  for (const hit of hits) {
    const element = document.createElement("div");
    element.className = "spec";
    element.dataset.layer = hit.source_type;
    element.innerHTML =
      `<span class="who">${esc(LAYER_NAME[hit.source_type] || hit.source_type)}` +
      ` · ${esc(hit.evidence_kind || "")}</span>` +
      `<button type="button" class="ttl" aria-expanded="false">${esc(hit.title)}</button>` +
      `<div class="full" hidden>${esc(hit.text)}` +
      `<span class="cid">${esc(hit.chunk_id)}</span></div>`;

    const toggle = element.querySelector(".ttl");
    const full = element.querySelector(".full");
    toggle.addEventListener("click", () => {
      full.hidden = !full.hidden;
      toggle.setAttribute("aria-expanded", String(!full.hidden));
    });
    listEl.appendChild(element);
  }
  specimensEl.hidden = false;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  recordQuestion(question);

  go.disabled = true;
  document.body.classList.add("is-thinking");
  form.setAttribute("aria-busy", "true");
  document.body.dataset.asked = "1";
  errorEl.hidden = true;

  // Local-only preview: no fetch(), EventSource, WebSocket or model call.
  rewriteEl.textContent = `LOCAL_PREVIEW > “${question}” > no request sent`;
  rewriteEl.hidden = false;
  renderAnswer(PREVIEW_ANSWER);
  resultWindow.hidden = false;
  setMinimized(resultWindow, false);
  metaEl.textContent = "OFFLINE PREVIEW · 0 NETWORK REQUESTS · $0.0000";
  metaEl.hidden = false;
  renderSpecimens(PREVIEW_HITS);

  go.disabled = false;
  document.body.classList.remove("is-thinking");
  form.removeAttribute("aria-busy");
});
