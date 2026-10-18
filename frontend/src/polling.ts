export type PollOutcome = "started" | "completed" | "failed" | "skipped";

export type PollTelemetry = {
  outcome: PollOutcome;
  duration_ms?: number;
  concurrent_poll_count: number;
  consecutive_failure_count: number;
};

type Observer = (event: PollTelemetry) => void;

/** Runs at most one refresh at a time, so a slow response cannot overwrite a newer poll. */
export class PollCoordinator {
  private inFlight = false;
  private consecutiveFailures = 0;

  async run<T>(operation: () => Promise<T>, observe: Observer): Promise<T | undefined> {
    if (this.inFlight) {
      observe({
        outcome: "skipped",
        concurrent_poll_count: 1,
        consecutive_failure_count: this.consecutiveFailures,
      });
      return undefined;
    }

    this.inFlight = true;
    const started = performance.now();
    observe({
      outcome: "started",
      concurrent_poll_count: 1,
      consecutive_failure_count: this.consecutiveFailures,
    });
    try {
      const result = await operation();
      this.consecutiveFailures = 0;
      observe({
        outcome: "completed",
        duration_ms: Math.round(performance.now() - started),
        concurrent_poll_count: 1,
        consecutive_failure_count: 0,
      });
      return result;
    } catch (error) {
      this.consecutiveFailures += 1;
      observe({
        outcome: "failed",
        duration_ms: Math.round(performance.now() - started),
        concurrent_poll_count: 1,
        consecutive_failure_count: this.consecutiveFailures,
      });
      throw error;
    } finally {
      this.inFlight = false;
    }
  }
}
