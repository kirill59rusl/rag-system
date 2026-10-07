import * as api from "./api.js";

const $ = (sel) => document.querySelector(sel);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

async function withBusy(form, fn) {
  const button = form.querySelector("button[type=submit]");
  button.disabled = true;
  const label = button.textContent;
  button.textContent = "…";
  try {
    await fn();
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

function showError(target, err) {
  target.replaceChildren(el("p", { class: "error" }, err.message));
}

// --- статус backend ---

async function checkHealth() {
  const status = $("#status");
  try {
    await api.health();
    status.textContent = "backend ok";
    status.className = "status ok";
  } catch {
    status.textContent = "backend недоступен";
    status.className = "status down";
  }
}

// --- загрузка ---

$("#upload-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const form = e.currentTarget;
  const out = $("#upload-result");
  const file = form.file.files[0];
  if (!file) return;

  withBusy(form, async () => {
    out.classList.remove("hidden", "error");
    out.textContent = `Загружаю ${file.name}…`;
    try {
      const doc = await api.upload(file);
      out.textContent = JSON.stringify(doc, null, 2);
      form.reset();
    } catch (err) {
      out.classList.add("error");
      out.textContent = err.message;
    }
  });
});

// --- вопрос / поиск ---

function sourceLocation(src) {
  const parts = [];
  if (src.section_path?.length) parts.push(src.section_path.join(" › "));
  else if (src.section) parts.push(src.section);
  if (src.page_number != null) {
    parts.push(
      src.page_end && src.page_end !== src.page_number
        ? `стр. ${src.page_number}–${src.page_end}`
        : `стр. ${src.page_number}`,
    );
  }
  return parts.join(" · ");
}

function renderSource(src, n) {
  return el(
    "details",
    { class: `source${src.used ? " used" : ""}`, id: n != null ? `src-${n}` : "" },
    el(
      "summary",
      {},
      n != null ? el("b", {}, `[${n}] `) : null,
      sourceLocation(src) || `chunk ${src.chunk_index}`,
      src.distance != null ? el("span", { class: "muted" }, ` · dist ${src.distance.toFixed(3)}`) : null,
    ),
    el("p", { class: "muted small" }, `doc ${src.doc_id} · chunk ${src.chunk_index}`),
    el("pre", {}, src.content ?? ""),
  );
}

function renderAnswer(res) {
  const claims = res.claims?.length
    ? el(
        "ul",
        { class: "claims" },
        res.claims.map((c) =>
          el(
            "li",
            {},
            c.text,
            " ",
            c.sources.map((id) => el("a", { href: `#src-${id}`, class: "ref" }, `[${id}]`)),
          ),
        ),
      )
    : null;

  return [
    el("div", { class: "answer" }, res.answer),
    claims,
    res.sources?.length ? el("h3", {}, "Источники") : null,
    res.sources?.map((s) => renderSource(s, s.id)),
  ];
}

$("#ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const form = e.currentTarget;
  const out = $("#answer");
  const params = {
    query: form.query.value.trim(),
    limit: form.limit.value,
    reranker_limit: form.reranker_limit.value,
  };
  if (!params.query) return;

  withBusy(form, async () => {
    out.replaceChildren(el("p", { class: "muted" }, "Думаю…"));
    try {
      if (form.retrieval_only.checked) {
        const chunks = await api.search(params);
        out.replaceChildren(
          el("h3", {}, `Найдено: ${chunks.length}`),
          ...chunks.map((c, i) => renderSource(c, i + 1)),
        );
      } else {
        out.replaceChildren(...renderAnswer(await api.generate(params)).flat().filter(Boolean));
      }
    } catch (err) {
      showError(out, err);
    }
  });
});

// Ctrl/Cmd+Enter отправляет вопрос
$("#ask-form textarea").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) e.currentTarget.form.requestSubmit();
});

checkHealth();
