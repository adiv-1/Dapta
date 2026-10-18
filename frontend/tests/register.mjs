// Lets Node load the TypeScript test entries, React sources, and Base44 Deno function. The hook
// maps Base44-only imports to local stubs and compiles project-owned TS/TSX with Vite's esbuild.
import { existsSync, readFileSync } from "node:fs";
import { registerHooks } from "node:module";
import { fileURLToPath } from "node:url";

import { transformSync } from "esbuild";

const stubs = {
  "npm:@base44/sdk": new URL("./stubs/base44-sdk.mjs", import.meta.url).href,
  "base44:runtime": new URL("./stubs/base44-runtime.mjs", import.meta.url).href,
};
const base44ClientStub = new URL("./stubs/base44-client.mjs", import.meta.url).href;
const srcDir = new URL("../src/", import.meta.url).href;
const testsDir = new URL("./", import.meta.url).href;
const base44FunctionsDir = new URL("../base44/functions/", import.meta.url).href;

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (stubs[specifier]) return { url: stubs[specifier], shortCircuit: true };
    if (context.parentURL?.startsWith(srcDir) && specifier.startsWith(".") && !/\.[cm]?[jt]sx?$/.test(specifier)) {
      for (const extension of [".ts", ".tsx"]) {
        const url = new URL(specifier + extension, context.parentURL);
        if (!existsSync(fileURLToPath(url))) continue;
        if (url.href === new URL("base44Client.ts", srcDir).href) return { url: base44ClientStub, shortCircuit: true };
        return { url: url.href, shortCircuit: true };
      }
    }
    return nextResolve(specifier, context);
  },
  load(url, context, nextLoad) {
    if ((url.startsWith(srcDir) || url.startsWith(testsDir) || url.startsWith(base44FunctionsDir))
        && /\.tsx?$/.test(url)) {
      const { code } = transformSync(readFileSync(fileURLToPath(url), "utf8"), {
        loader: url.endsWith(".tsx") ? "tsx" : "ts",
        format: "esm",
        jsx: "automatic",
        sourcefile: fileURLToPath(url),
        define: { "import.meta.env": "globalThis.__VITE_ENV__" },
      });
      return { format: "module", source: code, shortCircuit: true };
    }
    return nextLoad(url, context);
  },
});

globalThis.__VITE_ENV__ ??= { MODE: "test" };
