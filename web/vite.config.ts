import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist" },
  server: {
    // The API runs separately in dev; proxying keeps the frontend origin-clean
    // so there is no CORS configuration to get wrong.
    proxy: {
      "/search": "http://localhost:8000",
      "/answer": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});