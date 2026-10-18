import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { initializeSentry } from "./sentry";
import "./styles.css";

initializeSentry();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
