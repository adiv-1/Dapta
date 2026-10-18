import { defineConfig } from "vite";
import base44 from "@base44/vite-plugin";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [base44(), react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
