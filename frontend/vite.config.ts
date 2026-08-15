import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: { "/threads": "http://localhost:8000", "/health": "http://localhost:8000" } },
});
