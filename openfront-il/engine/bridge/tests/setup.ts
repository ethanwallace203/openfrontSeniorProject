import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

// Point every test at an isolated, throwaway data/ directory instead of the
// project's real one — types.ts's DATA_DIR is computed from this env var at
// module-load time, so it must be set before any test imports it.
const testDataDir = fs.mkdtempSync(path.join(os.tmpdir(), "openfront-il-test-"));
process.env.OPENFRONT_DATA_DIR = testDataDir;
