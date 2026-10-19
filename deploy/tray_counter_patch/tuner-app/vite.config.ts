import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" so the built dist/ can be served from any path (nginx subfolder, S3, the edge device).
export default defineConfig({
  base: "./",
  plugins: [react()],
  // in development the API and run files come from server/server.mjs (npm run dev:server, port 5181)
  server: { port: 5173, proxy: { "/api": "http://localhost:5181", "/runs": "http://localhost:5181" } },
});
