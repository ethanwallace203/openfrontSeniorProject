import * as path from "node:path";
import { defineConfig } from "vitest/config";

const openfrontRoot = path.resolve(__dirname, "engine/openfront-ai/openfront");

export default defineConfig({
  // Matches tsconfig.run.json's target/useDefineForClassFields: applied
  // here (not tsconfig.json) because Vite's esbuild transform, like tsx's,
  // only honors a tsconfig's compilerOptions for files matched by that
  // config's `include` — see tsconfig.run.json's own comment for why
  // vendored engine files need this to construct correctly.
  esbuild: {
    target: "es2020",
  },
  resolve: {
    alias: [
      { find: "resources", replacement: path.join(openfrontRoot, "resources") },
      { find: "src", replacement: path.join(openfrontRoot, "src") },
    ],
  },
  test: {
    include: ["engine/bridge/tests/**/*.test.ts"],
    exclude: ["engine/openfront-ai/**"],
    environment: "node",
    setupFiles: ["./engine/bridge/tests/setup.ts"],
  },
});
