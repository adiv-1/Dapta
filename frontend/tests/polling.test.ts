import assert from "node:assert/strict";
import { test } from "node:test";

import { PollCoordinator, type PollTelemetry } from "../src/polling.ts";

test("slow polls never overlap or let a skipped request replace current state", async () => {
  const coordinator = new PollCoordinator();
  const events: PollTelemetry[] = [];
  let finish!: (value: string) => void;
  const first = coordinator.run(() => new Promise<string>((resolve) => { finish = resolve; }), (event) => events.push(event));
  const skipped = await coordinator.run(async () => "stale", (event) => events.push(event));
  assert.equal(skipped, undefined);
  finish("current");
  assert.equal(await first, "current");
  assert.deepEqual(events.map((event) => event.outcome), ["started", "skipped", "completed"]);
});

test("poll telemetry counts consecutive failures and resets after recovery", async () => {
  const coordinator = new PollCoordinator();
  const events: PollTelemetry[] = [];
  for (let attempt = 0; attempt < 2; attempt += 1) {
    await assert.rejects(coordinator.run(async () => { throw new Error("synthetic"); }, (event) => events.push(event)));
  }
  await coordinator.run(async () => "ok", (event) => events.push(event));
  const failures = events.filter((event) => event.outcome === "failed");
  assert.deepEqual(failures.map((event) => event.consecutive_failure_count), [1, 2]);
  assert.equal(events.at(-1)?.consecutive_failure_count, 0);
});
