const chatScroll = document.getElementById("chatScroll");
const emptyState = document.getElementById("emptyState");
const composerForm = document.getElementById("composerForm");
const topicInput = document.getElementById("topicInput");
const sendBtn = document.getElementById("sendBtn");
const providerSelect = document.getElementById("providerSelect");
const countSelect = document.getElementById("countSelect");
const topicChipsEl = document.getElementById("topicChips");
const newChatBtn = document.getElementById("newChatBtn");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const mobileMenuBtn = document.getElementById("mobileMenuBtn");
const sidebar = document.getElementById("sidebar");
const sidebarOverlay = document.getElementById("sidebarOverlay");

const SUGGESTED_TOPICS = [
  "trailing zeros in a factorial",
  "successive percentage change",
  "alligation and mixtures",
  "time and work, negative work",
  "games and tournaments",
  "data sufficiency approach",
  "para jumble strategy",
  "critical reasoning assumptions",
];

let busy = false;

function init() {
  SUGGESTED_TOPICS.forEach((topic) => {
    const btn = document.createElement("button");
    btn.className = "topic-chip";
    btn.type = "button";
    btn.textContent = topic;
    btn.addEventListener("click", () => {
      topicInput.value = topic;
      closeMobileSidebar();
      composerForm.requestSubmit();
    });
    topicChipsEl.appendChild(btn);
  });

  checkHealth();
  composerForm.addEventListener("submit", onSubmit);
  newChatBtn.addEventListener("click", resetChat);
  mobileMenuBtn.addEventListener("click", () => sidebar.classList.toggle("open") || sidebarOverlay.classList.toggle("open"));
  sidebarOverlay.addEventListener("click", closeMobileSidebar);
}

function closeMobileSidebar() {
  sidebar.classList.remove("open");
  sidebarOverlay.classList.remove("open");
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error();
    const data = await res.json();
    statusDot.className = "status-dot ok";
    statusText.textContent = `Ready — ${data.qa_pairs} linked pairs indexed`;
  } catch {
    statusDot.className = "status-dot err";
    statusText.textContent = "Backend unreachable";
  }
}

function resetChat() {
  chatScroll.innerHTML = "";
  chatScroll.appendChild(emptyState);
  emptyState.style.display = "";
  topicInput.value = "";
  topicInput.focus();
}

function scrollToBottom() {
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

async function onSubmit(e) {
  e.preventDefault();
  const topic = topicInput.value.trim();
  if (!topic || busy) return;

  emptyState.style.display = "none";
  busy = true;
  sendBtn.disabled = true;

  const userRow = el(`
    <div class="message-row user"><div class="bubble-user"></div></div>
  `);
  userRow.querySelector(".bubble-user").textContent = topic;
  chatScroll.appendChild(userRow);

  const assistantRow = el(`
    <div class="message-row">
      <div class="assistant-block">
        <div class="thinking"><span></span><span></span><span></span></div>
      </div>
    </div>
  `);
  chatScroll.appendChild(assistantRow);
  scrollToBottom();

  topicInput.value = "";

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        topic,
        n: parseInt(countSelect.value, 10),
        provider: providerSelect.value,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Something went wrong.");
    }
    renderQuiz(assistantRow.querySelector(".assistant-block"), data);
  } catch (err) {
    renderError(assistantRow.querySelector(".assistant-block"), err.message);
  } finally {
    busy = false;
    sendBtn.disabled = false;
    scrollToBottom();
  }
}

function renderError(container, message) {
  container.innerHTML = "";
  container.appendChild(el(`<div class="error-box">${escapeHtml(message)}</div>`));
}

function renderQuiz(container, data) {
  container.innerHTML = "";

  data.questions.forEach((q, qi) => {
    const card = el(`
      <div class="quiz-card">
        <div class="quiz-question"></div>
        <div class="quiz-options"></div>
        <div class="reveal">
          <div><span class="label">Correct answer:</span> <span class="ans-text"></span></div>
          <div style="margin-top:6px"><span class="label">Shortcut used:</span> <span class="sc-text"></span></div>
        </div>
      </div>
    `);
    card.querySelector(".quiz-question").textContent = `${qi + 1}. ${q.question}`;

    const optsEl = card.querySelector(".quiz-options");
    const letters = ["A", "B", "C", "D"];
    q.options.forEach((opt, oi) => {
      const btn = el(`
        <button type="button" class="option-btn">
          <span class="opt-letter"></span><span class="opt-text"></span>
        </button>
      `);
      btn.querySelector(".opt-letter").textContent = letters[oi] + ".";
      btn.querySelector(".opt-text").textContent = opt;
      btn.addEventListener("click", () => {
        const allBtns = optsEl.querySelectorAll(".option-btn");
        allBtns.forEach((b) => (b.disabled = true));
        if (opt === q.correct_answer) {
          btn.classList.add("correct");
        } else {
          btn.classList.add("incorrect");
          allBtns.forEach((b) => {
            if (b.querySelector(".opt-text").textContent === q.correct_answer) {
              b.classList.add("correct");
            }
          });
        }
        card.querySelector(".reveal").classList.add("show");
      });
      optsEl.appendChild(btn);
    });

    card.querySelector(".ans-text").textContent = q.correct_answer;
    card.querySelector(".sc-text").textContent = q.shortcut_used;

    if (q.verified === false) {
      const warn = document.createElement("div");
      warn.className = "verify-warn";
      warn.textContent = "Could not be fully verified -- double-check this one.";
      card.querySelector(".reveal").appendChild(warn);
    }

    container.appendChild(card);
  });

  if (data.retrieved_context && data.retrieved_context.length) {
    const details = el(`
      <details class="sources">
        <summary></summary>
        <ul class="sources-list"></ul>
      </details>
    `);
    details.querySelector("summary").textContent = `Sources (${data.retrieved_context.length})`;
    const list = details.querySelector(".sources-list");
    data.retrieved_context.forEach((c) => {
      const li = document.createElement("li");
      const tag = c.role && c.role.startsWith("linked") ? c.role.replace("linked_", "linked ") : "retrieved";
      const scoreStr = typeof c.score === "number" ? c.score.toFixed(3) : "—";
      li.innerHTML = `<span class="src-tag">${escapeHtml(tag)}</span>${escapeHtml(c.source_file)} (p${c.page_start}) &middot; ${scoreStr}`;
      list.appendChild(li);
    });
    container.appendChild(details);
  }
}

function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}

init();
