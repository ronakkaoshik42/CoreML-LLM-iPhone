# Project agent rules

- Never stage or commit model blobs, including `.mlpackage`, `.mlmodelc`, and
  `.bin` files.
- Do not change the app bundle ID.
- Do not delete, move, or replace model files on the Mac or iPhone.
- Do not commit DerivedData, `.xcresult`, app containers, device logs, signing
  secrets, provisioning profiles, or certificates.
- Preserve the existing 4B/8B sideload scripts and their current bundle-ID
  alignment.
- Keep edits small and reviewable. Inspect the working tree before editing and
  stage files by explicit path.
- Prefer measuring before optimizing. Change one performance variable at a
  time and retain a comparable baseline.
- Treat the wrapper-level `[RESULT]` line (currently the second result line for
  a successful generation) as the source of truth.
- For 8B benchmarks, remind the user to keep the physical iPhone plugged in and
  unlocked.
- Validate 4B text before expanding a benchmark or optimization to 8B/image.
