import { Component, type ErrorInfo, type ReactNode } from "react";
import * as Sentry from "@sentry/react";

export class AppErrorBoundary extends Component<{ children: ReactNode }, { error?: Error }> {
  state: { error?: Error } = {};

  static getDerivedStateFromError(error: Error) { return { error }; }

  componentDidCatch(error: Error, info: ErrorInfo) {
    Sentry.captureException(error, {
      tags: { "dapta.error_category": "react_render" },
      contexts: { react: { component_stack: info.componentStack || "unavailable" } },
    });
    console.error("Dapta UI recovered from a render error", error, info);
  }

  render() {
    if (this.state.error) {
      return <main className="recover-screen"><div><strong>Dapta is still running.</strong><p>The live view hit a temporary display error. Your backend job and artifacts were not interrupted.</p><button onClick={() => this.setState({ error: undefined })}>Restore live view</button></div></main>;
    }
    return this.props.children;
  }
}
