/*
 * frontends/web/static/app.js
 * ---------------------------
 * The whole client. It holds no state the server does not: every action posts
 * to the API and redraws from the `state` payload that comes back, so a reload
 * (or a second tab) shows the same conversation.
 */

const $ = (id) => document.getElementById(id);

const el = {
  cwd: $("cwd"),
  threads: $("threads"),
  transcript: $("transcript"),
  approval: $("approval"),
  approvalCalls: $("approval-calls"),
  feedback: $("feedback"),
  prompt: $("prompt"),
  send: $("send"),
  statusPill: $("status-pill"),
  tokens: $("tokens"),
  notes: $("notes"),
  goal: $("goal"),
  overseerStatus: $("overseer-status"),
  overseerStart: $("overseer-start"),
  overseerStop: $("overseer-stop"),
  autoApprove: $("auto-approve"),
  steps: $("steps"),
  fileFilter: $("file-filter"),
  fileTree: $("file-tree"),
  fileContent: $("file-content"),
  openFile: $("open-file"),
  saveFile: $("save-file"),
  toast: $("toast"),
};

let state = null;
let busy = false;
let allFiles = [];
let openPath = null;

/* --- helpers ------------------------------------------------------------ */

let toastTimer = null;
function toast(message) {
  el.toast.textContent = message;
  el.toast.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.toast.classList.add("hidden"), 5000);
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (_) { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return res.json();
}

function post(path, body) {
  return api(path, { method: "POST", body: JSON.stringify(body || {}) });
}

/*
 * Stream an SSE response from a POST. EventSource cannot POST, so the frames
 * are parsed by hand: split on the blank line that terminates each one.
 */
async function stream(path, body, handlers) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (_) { /* non-JSON error body */ }
    throw new Error(detail);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let split;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      let event = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      const handler = handlers[event];
      if (handler) handler(JSON.parse(data));
    }
  }
}

function setBusy(value) {
  busy = value;
  el.send.disabled = value;
  el.prompt.disabled = value;
  el.statusPill.textContent = value ? "thinking…" : (state && state.ready ? "ready" : "not configured");
  el.statusPill.className = "pill" + (value ? "" : state && state.ready ? " ok" : " bad");
}

/* --- rendering ---------------------------------------------------------- */

function callNode(call) {
  const node = document.createElement("div");
  node.className = "call";
  const name = document.createElement("div");
  name.className = "name";
  name.textContent = call.name;
  const args = document.createElement("pre");
  args.textContent = JSON.stringify(call.args, null, 2);
  node.append(name, args);
  return node;
}

function messageNode(msg) {
  const node = document.createElement("div");
  node.className = `msg ${msg.role}`;
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = msg.role === "tool" ? `tool · ${msg.name || ""}` : msg.role;
  const body = document.createElement("div");
  body.className = "body";
  body.textContent = msg.content;
  node.append(who, body);
  if (msg.tool_calls && msg.tool_calls.length) {
    const calls = document.createElement("div");
    calls.className = "calls";
    msg.tool_calls.forEach((c) => calls.append(callNode(c)));
    node.append(calls);
  }
  return node;
}

function atBottom() {
  const t = el.transcript;
  return t.scrollHeight - t.scrollTop - t.clientHeight < 80;
}

function scrollToEnd() {
  el.transcript.scrollTop = el.transcript.scrollHeight;
}

function renderTranscript() {
  const stick = atBottom();
  el.transcript.replaceChildren();
  if (!state.messages.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No messages yet. Ask the agent to do something.";
    el.transcript.append(empty);
  } else {
    state.messages.forEach((m) => el.transcript.append(messageNode(m)));
  }
  if (stick) scrollToEnd();
}

function renderApproval() {
  const calls = state.pending_tool_calls;
  el.approval.classList.toggle("hidden", calls.length === 0);
  el.approvalCalls.replaceChildren();
  calls.forEach((call) => {
    el.approvalCalls.append(callNode(call));
    if (call.justification) {
      const why = document.createElement("p");
      why.className = "justification";
      why.textContent = call.justification;
      el.approvalCalls.append(why);
    }
  });
}

function renderOverseer() {
  const o = state.overseer;
  el.overseerStart.disabled = o.active;
  el.overseerStop.disabled = !o.active;
  if (!o.active) {
    el.overseerStatus.textContent = o.step ? `Stopped after ${o.step} step(s).` : "Idle.";
  } else {
    el.overseerStatus.textContent = `Step ${o.step}: ${o.last_instruction || "planning…"}`;
  }
}

function render(next) {
  state = next;
  el.cwd.textContent = state.cwd;
  el.tokens.textContent = `${state.token_count.toLocaleString()} tokens`;
  if (!busy) setBusy(false);
  renderTranscript();
  renderApproval();
  renderOverseer();
}

async function refreshThreads() {
  const { current, threads } = await api("/api/threads");
  el.threads.replaceChildren();
  threads.forEach((t) => {
    const li = document.createElement("li");
    if (t.thread_id === current) li.classList.add("active");
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = t.last_human || `${t.thread_id.slice(0, 8)} (empty)`;
    label.title = `${t.thread_id} · ${t.message_count} messages`;
    label.onclick = () => guarded(async () => {
      render(await post("/api/threads/switch", { thread_id: t.thread_id }));
      refreshThreads();
    });
    const x = document.createElement("button");
    x.className = "x";
    x.textContent = "×";
    x.title = "Delete this conversation";
    x.onclick = (ev) => {
      ev.stopPropagation();
      if (!confirm("Delete this conversation? This cannot be undone.")) return;
      guarded(async () => {
        render(await post("/api/threads/delete", { thread_id: t.thread_id }));
        refreshThreads();
      });
    };
    li.append(label, x);
    el.threads.append(li);
  });
}

/* --- actions ------------------------------------------------------------ */

async function guarded(fn) {
  try {
    await fn();
  } catch (err) {
    toast(err.message);
  }
}

async function runTurn(path, body) {
  if (busy) return;
  setBusy(true);
  // A live bubble the tokens stream into; the final `state` frame replaces it
  // with the transcript the server actually persisted.
  const live = messageNode({ role: "assistant", content: "" });
  const body_ = live.querySelector(".body");
  el.transcript.append(live);
  scrollToEnd();
  try {
    await stream(path, body, {
      token: (d) => {
        const stick = atBottom();
        body_.textContent += d.text;
        if (stick) scrollToEnd();
      },
      error: (d) => toast(d.detail),
      state: (d) => render(d),
    });
  } catch (err) {
    live.remove();
    toast(err.message);
  } finally {
    setBusy(false);
    refreshThreads();
  }
}

$("composer").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const prompt = el.prompt.value.trim();
  if (!prompt || busy) return;
  el.prompt.value = "";
  el.prompt.style.height = "auto";
  el.transcript.append(messageNode({ role: "user", content: prompt }));
  scrollToEnd();
  runTurn("/api/chat", { prompt });
});

el.prompt.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    $("composer").requestSubmit();
  }
});

el.prompt.addEventListener("input", () => {
  el.prompt.style.height = "auto";
  el.prompt.style.height = `${Math.min(el.prompt.scrollHeight, 200)}px`;
});

$("approve").onclick = async () => {
  await runTurn("/api/tools/approve", {});
  resumeOverseerIfParked();
};

$("send-feedback").onclick = async () => {
  const feedback = el.feedback.value.trim();
  if (!feedback) return toast("Type a reply first.");
  el.feedback.value = "";
  await runTurn("/api/tools/feedback", { feedback });
  resumeOverseerIfParked();
};

// Denying throws away the turn the Overseer's instruction produced, so the
// run cannot meaningfully continue from it - stop rather than re-instruct.
$("deny").onclick = () => guarded(async () => {
  render(await post("/api/tools/deny"));
  if (state.overseer.active) {
    render(await post("/api/overseer/stop"));
    toast("Autopilot stopped: the tool call was denied.");
  }
});

$("new-thread").onclick = () => guarded(async () => {
  render(await post("/api/threads/new"));
  refreshThreads();
});

$("undo-last").onclick = () => guarded(async () => render(await post("/api/history/undo_last")));
$("undo-first").onclick = () => guarded(async () => render(await post("/api/history/undo_first")));
$("clear-history").onclick = () => {
  if (!confirm("Clear every message in this conversation?")) return;
  guarded(async () => render(await post("/api/history/clear")));
};

/* --- autopilot ---------------------------------------------------------- */

/*
 * The loop lives here rather than on the server so Stop takes effect between
 * steps, and so a run can be driven with tools auto-approved (what the CLI's
 * --goal does) or with a human in the loop.
 *
 * A turn almost always ends on a tool call, so settling those is part of the
 * step, not the end of the run: with auto-approve on they are approved until
 * the agent stops asking, and with it off the run parks and the approval
 * buttons restart it.
 */
let overseerRunning = false;

const autoApprove = () => el.autoApprove.checked;
const stepLimit = () => Math.max(0, parseInt(el.steps.value, 10) || 0);

/*
 * Settle every tool call the agent is waiting on. Returns false when the loop
 * must not continue: parked for a human, or stopped mid-approval.
 */
const MAX_AUTO_APPROVALS_PER_STEP = 25;

async function settleToolCalls() {
  let approvals = 0;
  while (state.overseer.active && state.pending_tool_calls.length) {
    if (approvals >= MAX_AUTO_APPROVALS_PER_STEP) {
      // An agent stuck in a tool loop would otherwise run unattended forever.
      render(await post("/api/overseer/stop"));
      toast(`Autopilot stopped: ${MAX_AUTO_APPROVALS_PER_STEP} tool calls in one step.`);
      return false;
    }
    if (!autoApprove()) {
      el.overseerStatus.textContent =
        `Step ${state.overseer.step}: waiting for you to approve a tool call.`;
      return false;
    }
    await runTurn("/api/tools/approve", {});
    approvals += 1;
  }
  return state.overseer.active;
}

async function overseerLoop() {
  if (overseerRunning) return;
  overseerRunning = true;
  try {
    // Pick up where a parked run left off before asking for a new instruction.
    if (!(await settleToolCalls())) return;

    while (state.overseer.active) {
      const limit = stepLimit();
      if (limit && state.overseer.step >= limit) {
        render(await post("/api/overseer/stop"));
        toast(`Autopilot stopped: ${limit}-step limit reached.`);
        return;
      }
      let step;
      try {
        step = await post("/api/overseer/step");
      } catch (err) {
        toast(err.message);
        render(await api("/api/state"));
        return;
      }
      if (!state.overseer.active) return; // Stop was pressed while it thought
      el.transcript.append(messageNode({ role: "user", content: step.instruction }));
      scrollToEnd();
      await runTurn("/api/chat", { prompt: step.instruction });
      if (!(await settleToolCalls())) return;
    }
  } finally {
    overseerRunning = false;
  }
}

/* Restart a parked run once the human has dealt with the tool call. */
function resumeOverseerIfParked() {
  if (state.overseer.active && !overseerRunning) overseerLoop();
}

el.overseerStart.onclick = () => guarded(async () => {
  const goal = el.goal.value.trim();
  if (!goal) return toast("Give the Overseer a goal first.");
  render(await post("/api/overseer/start", { goal }));
  overseerLoop();
});

el.overseerStop.onclick = () => guarded(async () => render(await post("/api/overseer/stop")));

/* --- notes -------------------------------------------------------------- */

$("save-notes").onclick = () => guarded(async () => {
  await post("/api/notes", { content: el.notes.value });
  toast("Notes saved.");
});

/* --- files -------------------------------------------------------------- */

function renderFiles() {
  const needle = el.fileFilter.value.trim().toLowerCase();
  el.fileTree.replaceChildren();
  allFiles
    .filter((p) => !needle || p.toLowerCase().includes(needle))
    .slice(0, 500)
    .forEach((path) => {
      const li = document.createElement("li");
      li.textContent = path;
      li.title = path;
      if (path === openPath) li.classList.add("active");
      li.onclick = () => guarded(async () => {
        const file = await api(`/api/files/read?path=${encodeURIComponent(path)}`);
        openPath = path;
        el.openFile.textContent = path;
        el.fileContent.value = file.content;
        el.saveFile.disabled = false;
        renderFiles();
      });
      el.fileTree.append(li);
    });
}

el.fileFilter.addEventListener("input", renderFiles);

el.saveFile.onclick = () => guarded(async () => {
  if (!openPath) return;
  const { result } = await post("/api/files/write", { path: openPath, content: el.fileContent.value });
  toast(result);
});

document.querySelectorAll(".tab").forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll(".view").forEach((v) => {
      v.classList.toggle("active", v.id === `view-${tab.dataset.view}`);
    });
    if (tab.dataset.view === "files") {
      guarded(async () => {
        allFiles = (await api("/api/files")).files;
        renderFiles();
      });
    }
  };
});

/* --- boot --------------------------------------------------------------- */

guarded(async () => {
  render(await api("/api/state"));
  el.notes.value = (await api("/api/notes")).content;
  await refreshThreads();
  if (!state.ready) {
    toast("The model is not configured. Set GOOGLE_API_KEY and MODEL_ID, then reload.");
  }
});
