// `npm run release`: a production build that refuses to ship unsigned.
//
// Sets CADENCE_REQUIRE_SIGNING=1 for scripts/sign.ps1 (so every binary Tauri
// and scripts/prepare-runtimes.ps1 sign must end up with a trusted signature),
// then runs `tauri build`. A plain `npm run tauri build` stays usable for
// local, unsigned test builds and warns instead.
//
// It also gives sign.ps1 a failure log (CADENCE_SIGN_FAILURE_LOG) and fails if
// anything landed in it. makensis runs sign.ps1 for the uninstaller through
// !uninstfinalize and ignores its exit code, so without this check a failed
// uninstaller signature would still produce a "successful" release.
//
// CI release builds sign through SignPath instead; see
// .github/workflows/release.yml and docs/release.md.
import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const failureLog = join(tmpdir(), `cadence-sign-failures-${process.pid}.log`);
rmSync(failureLog, { force: true });

const env = {
  ...process.env,
  CADENCE_REQUIRE_SIGNING: "1",
  CADENCE_SIGN_FAILURE_LOG: failureLog,
};
const result = spawnSync("npx", ["tauri", "build", ...process.argv.slice(2)], {
  stdio: "inherit",
  env,
  shell: process.platform === "win32",
});

if (existsSync(failureLog)) {
  const failures = readFileSync(failureLog, "utf8").trim();
  rmSync(failureLog, { force: true });
  if (failures) {
    console.error(
      "\nrelease: signing failed for at least one file, so this build must not be released:\n" +
        failures +
        "\n(The uninstaller is signed inside makensis, which does not stop on a signing failure.)",
    );
    process.exit(1);
  }
}
process.exit(result.status ?? 1);
