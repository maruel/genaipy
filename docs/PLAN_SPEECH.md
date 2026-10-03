# Reusable local speech backends with verified genai integration

Update genaipy for the current genai API, establish deterministic testing, verify
llama.cpp speech support, and move reusable speech runtimes out of gomode.
Dependency rolls remain owner-managed; local integration uses an ignored workspace.

## Phase 1 — reusable-speech: Shared ASR and TTS used by gomode

- **Scope:** Reusable HTTP transcription/synthesis and KittenTTS runtime in
  genaipy; gomode retains voice session orchestration and consumes shared APIs.
- **Verify:** Deterministic speech success, failure, cancellation and streaming
  tests; genaipy gates; affected gomode tests and its documented gates.
