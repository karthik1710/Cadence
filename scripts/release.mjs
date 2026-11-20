// `npm run release`: a production build that refuses to ship unsigned.
//
// Sets CADENCE_REQUIRE_SIGNING=1 for scripts/sign.ps1 (so every binary Tauri
// and scripts/prepare-runtimes.ps1 sign must end up with a trusted signature),
// then runs `tauri build`. A plain `npm run tauri build` stays usable for
// local, unsigned test builds and warns instead.
import { spawnSync } from "node:child_process";

const env = { ...process.env, CADENCE_REQUIRE_SIGNING: "1" };
const result = spawnSync("npx", ["tauri", "build", ...process.argv.slice(2)], {
  stdio: "inherit",
  env,
  shell: process.platform === "win32",
});
process.exit(result.status ?? 1);
