// A jsdom document for rendering the real React components in Node.
import { JSDOM } from "jsdom";

const dom = new JSDOM("<!doctype html><html><body><div id=\"root\"></div></body></html>", {
  url: "https://dapta.test/",
  pretendToBeVisual: true,
});

const globals = globalThis as Record<string, unknown>;
for (const name of ["window", "document", "HTMLElement", "Node", "Element", "MutationObserver", "getComputedStyle",
  "requestAnimationFrame", "cancelAnimationFrame", "HTMLInputElement", "Event", "CustomEvent"]) {
  if (!(name in globals) || name === "window" || name === "document") {
    globals[name] = name === "window" ? dom.window : (dom.window as unknown as Record<string, unknown>)[name];
  }
}
globals.IS_REACT_ACT_ENVIRONMENT = true;

export function container(): HTMLElement {
  const root = document.createElement("div");
  document.body.appendChild(root);
  return root;
}
