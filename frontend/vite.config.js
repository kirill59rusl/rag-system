import { defineConfig } from "vite";

const API_URL = process.env.API_URL ?? "http://localhost:8000";

// Всё, что начинается с /api, уходит на backend без префикса — так не нужен CORS.
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: API_URL,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
