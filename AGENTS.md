# AGENTS for Talk To Me

## Purpose
This document defines what an automated agent (or human collaborator following agent-style prompts) may do in the Talk To Me repository and how it should behave. It is tailored to the current project (a Spring Boot ONNX-based TTS prototype) and supports the workflow where the agent prepares changes locally and the user reviews and commits them manually (Option B).

## Scope
- Goal: help make targeted code changes, run and validate the project locally, create patches or change sets, and provide clear PR/commit descriptions for the human reviewer.
- Not in scope: automatically pushing commits to remote branches, publishing model files, exfiltrating secrets, or making production deployments.

## Agent persona & permissions
- Operates on the repository in the local workspace only.
- May edit files in-place in the working tree on branch `main` (or whichever branch is checked out locally), but must never push to the remote or open PRs automatically.
- Must produce a patch artifact (diff/patch file) or an explicit list of changed files + suggested commit message and PR description for the human to review and commit.

## Environment & how to run
- Project type: Spring Boot (Kotlin) application built with Gradle (wrapper present).
- Useful commands (run locally; agent should include these in instructions rather than executing remote actions):
  - Start application: ./gradlew bootRun
  - Run tests: ./gradlew test
  - Build: ./gradlew build
- Default configuration is in `src/main/resources/application.properties`.
  - Default basic auth: user:password
  - Storage path: `storage`
  - ONNX GPU enabled by default: `onnx.use-gpu=true`
- Required model files (not included): see `README.md` and `tts.models.pocket-tts` properties. Agent must not attempt to upload or fetch model files from private sources.

## Repo most important files (key files & symbols)
- `README.md` — project summary and model requirements
- `src/main/resources/application.properties` — default config and credentials
- `src/main/kotlin/de/dbaelz/ttm/controller/TtsController.kt` — REST API definitions
- `src/main/kotlin/de/dbaelz/ttm/service/DefaultTtsService.kt` — job lifecycle and executor wiring
- `src/main/kotlin/de/dbaelz/ttm/tts/pocket/PocketTtsExecutor.kt` — Pocket TTS implementation and heavy ONNX logic
- `src/main/kotlin/de/dbaelz/ttm/tts/pocket/SentencePieceTokenizer.kt` — tokenizer wrapper for SentencePiece used with Pocket TTS
- `src/main/kotlin/de/dbaelz/ttm/TalkToMeApplication.kt` — main entry point

## Primary API surface (from controller & tests)
- POST /api/tts
  - Request JSON: {"text": "...", "config": {...}?, "engine": "pocket"?}
  - Responses:
    - 202 Accepted + JSON body with job object on success (job fields include `id`, `text`, optional `engine`)
    - 400 Bad Request for invalid engine or missing text (body contains {"error": "INVALID_ENGINE", "message": "...", "allowedEngines": [...]})
    - 401 Unauthorized when missing/invalid basic auth
- GET /api/tts/jobs
  - 200 OK + JSON array of jobs
  - 401 Unauthorized when missing/invalid basic auth
- GET /api/tts/jobs/{id}
  - 200 OK + single job JSON or 404 Not Found
- GET /api/tts/files/{id}
  - 200 OK + Content-Type: audio/wav with bytes
  - 404 Not Found if missing

## Credentials for local runs
- Default basic auth from `application.properties`:
  - username: user
  - password: password
- Tests use the base64 header: Basic <Base64(user:password)>

## Core rules
The agent must follow these rules when making changes, running tests, and preparing patches for review. The agent should prioritize small, targeted changes that are easy to review and validate locally.

### Allowed actions (what the agent may do)
- Run local builds and tests in the developer environment and report results.
- Modify source and test files in the working tree to implement fixes or features.
- Generate a patch (uncommitted diff) capturing all modifications.
- Produce a suggested commit message and a concise PR description template.
- Create or update documentation files such as `README.md` or `AGENTS.md` itself.
- Provide concrete example HTTP requests for testing (curl/bruno snippets) and expected outputs.

### Forbidden actions (must never do)
- Push commits or open PRs on remote repositories without explicit human approval.
- Upload, download, or exfiltrate model files, voice WAVs, or any large binary assets to external services.
- Commit secrets (API keys, passwords) to the repository. If secrets are required for testing, document them as local-only and use environment variables or local config ignored in .gitignore.

### Patch workflow
Make changes and the user commits manually. When the agent makes changes, follow this exact handoff procedure:

1. Make the edits in the working tree. Do not commit or push.
2. Produce information about your changes and short description why you did them for the human reviewer
3. Provide a short checklist for the reviewer (how to reproduce tests, run the app, and verify the changes).
4. Optionally, include an annotated list of changed files with line ranges and a short explanation for each change.

### Prompt templates the agent should use internally
- Run tests and summarize failures:
  - "Run `./gradlew test`. If tests fail, present failing test names, stack traces, and a minimal plan to fix them. Prioritize small, targeted fixes. Create a patch with changes and the suggested commit message."
- Implement feature or fix:
  - "Implement <short feature description>. Make minimal code and test changes. Run `./gradlew test`. If green, produce `agent-changes.patch`, suggested commit message, PR description, and a reviewer checklist."
- Generate a sample audio using local runtime (manual verification):
  - "Start app `./gradlew bootRun` and send POST /api/tts with basic auth `user:password`. Expect 202 and a job id. Poll job or fetch file once complete." 

### Failure handling and retries
- If a command fails (build/test/run), the agent must:
  1. Capture and report the full error output.
  2. Attempt 1-2 local fixes if they are small and low-risk (typos, imports, obvious test assertions) and re-run tests.
  3. If not resolvable, produce a clear failure report with next steps and return the uncommitted patch (if any). Do not attempt destructive changes.

### Privacy & safety
- Never upload or share model binaries or sample audio to external services.
- Treat `application.properties` defaults as local dev defaults only. If secrets appear in the workspace, notify the reviewer and redact them from any patch or report.

### Onboarding checklist for this repository (what to run first)
- Confirm Java/Gradle environment available.
- Run `./gradlew test` to ensure the environment is healthy.
- Start the app with `./gradlew bootRun` and verify the API responds locally.
- Use provided Bruno collection (`bruno/bruno.json`) or `curl` to exercise `POST /api/tts` and `GET /api/tts/files/{id}`.
