const BASE = "/api";

async function request(path, options = {}) {
  const res = await fetch(BASE + path, options);
  const body = await res.text();
  let data;
  try {
    data = body ? JSON.parse(body) : null;
  } catch {
    data = body;
  }
  if (!res.ok) {
    const detail = data?.detail ?? data ?? res.statusText;
    throw new Error(`${res.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
  }
  return data;
}

function query(params) {
  return new URLSearchParams(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== ""),
  ).toString();
}

export const health = () => request("/health");

export function upload(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/documents/upload", { method: "POST", body: form });
}

export const search = (params) =>
  request(`/retrieval/search?${query(params)}`, { method: "POST" });

export const generate = (params) =>
  request(`/llm/generate?${query(params)}`, { method: "POST" });
