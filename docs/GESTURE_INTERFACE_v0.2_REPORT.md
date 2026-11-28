# FACEVISION — Gesture Interface v0.2

Technical report — 28 November 2026

> Public edition of the local Codex report. Publication followed the experiment. Internal backup paths and original measurements describe the local environment, not files distributed in this repository. Raw local logs and enrollment-related frame counts are not published. The public summary is linked below.

## Delivered behavior

The default application is now a small touchless launcher. It exposes exactly four entries, in order:

1. ChatGPT — `https://chatgpt.com/`
2. Browser — `https://www.google.com/`
3. YouTube — `https://www.youtube.com/`
4. GitHub — `https://github.com/`

Browser opens Google following the user's final correction. The earlier blank-tab implementation and its Windows registry workaround have been removed. Every destination now uses the same standard `webbrowser.open(fixed_url, new=2)` path, dispatched from a worker thread. The index is validated against the four-entry tuple; no arbitrary URL or shell command is accepted.

Open palm displays the menu with ChatGPT selected. Victory advances one entry and wraps GitHub to ChatGPT. Thumbs up launches the current entry and hides the menu; thumbs down cancels it. The menu also closes on inactivity, lost hand association, ambiguity or stale input. A small camera-status window remains available with an Exit button and native close button. Ctrl+Q exits when a FaceVision window has focus; Esc cancels the menu, and G toggles hand processing.

Current UI text, help, runtime messages and release documentation are English. The existing dark four-card layout, cyan selection, bundled vector icons and short instructions are kept simple. No animation, gradient, downloaded UI asset or extra dependency has been added. Historical reports remain preserved with their original versions and are not part of the new English release documentation.

## Architecture and preservation

Work is isolated on `experiment/gesture-interface`. Gesture Language v0.1 remains at commit `1b3bd81` on `experiment/gesture-language`; its source archive and complete package were copied into `backup/gesture-interface-baseline-1b3bd81` before implementation. No changes were pushed or published to GitHub.

`app.run()` remains the owner of the existing camera and face-processing loop. It now accepts an optional result observer, stop event and command queue. Without an observer, the original developer camera interface and controls remain available. With an observer, that same loop runs on a daemon camera thread and skips OpenCV GUI calls. It never opens an additional camera. The MediaPipe worker remains the existing worker, using the same frames.

`launcher_ui.py` owns one Tk root and one initially withdrawn menu Toplevel, all on the main thread. The camera thread publishes an immutable `FrameState` into a locked one-item mailbox. It contains confirmed hand results and a few counters/flags, not image arrays. Tk polls at a requested 20 ms interval. Rendering changes only when menu state, selection, instructions or camera status changes. The GUI clock is sampled after reading the mailbox to avoid treating a newly published result as a future timestamp.

`gesture_interface.py` contains a pure menu state machine and an asynchronous action dispatcher. No old v0.1 sequence recognizer or browser dispatcher is instantiated in launcher mode; an engine-boundary test verifies this. `--developer` retains v0.1 behavior, while `--developer --no-gesture-language` provides the original diagnostic view without its sequence action. Existing time-limited face benchmark CLI invocations continue using the developer path unless `--interface` is explicit.

YuNet, SFace, the MediaPipe model, the local owner profile, camera backends/resolutions and dependency pins are preserved. Some legacy console/help/docstring text was translated into English. AST comparison of `face_vision.py`, `tracking.py`, `metrics.py` and `gesture_language.py`, with string literals normalized, confirms that their non-text structure is unchanged. Model hashes and profile-preservation validation are recorded in [public validation summary](../benchmarks/gesture_interface_v0.2_summary.json).

The local owner profile remains unchanged. Profiles, private captures, model binaries and local experimental results are excluded from Git. This report and its numeric results are stored in the ignored results directory. The new distributable contains public models and English documentation, but no biometric profile or private camera capture.

## Confirmation and release safeguards

Default confirmation is **300 ms**, inactivity timeout **10 s**, and post-launch cooldown **5 s**. They can be changed with `--gesture-hold-ms`, `--menu-timeout` and `--launch-cooldown`.

Commands use the already validated hand output: generation-qualified track ID, capture timestamp, continuous confirmation onset and ambiguity flag. Repeated GUI reads of the same inference do not accumulate confirmation time. The same continuously tracked hand owns the active menu. Switching hands, losing the hand, ambiguous matching, reversed timestamps, gaps over 250 ms or results older than 250 ms close the menu. Two simultaneous open palms do not arbitrarily choose a command hand.

Each accepted gesture is latched. Holding Victory cannot repeatedly advance selection. Confirmed changes to another hand pose release that latch. The first usability test showed that relaxing the hand into an unclassified position did not release it, making repeated Victory awkward. The corrected implementation adds a release-only signal: model category None at confidence at least 0.70, continuously observed for the configured hold duration, can release the latch. A low-confidence Victory or a brief classification dropout cannot do so. None remains unclassified and cannot itself select or launch anything. The hint changes from Relax your hand, then repeat Victory to Ready for the next gesture.

After a launch, the menu closes and a new open-palm activation is required after cooldown. A palm held through cooldown cannot carry across its boundary. G cannot bypass cooldown. Enrollment disables commands. The launcher does not add a hand-to-owner identity binding; face ownership and hand commands remain separate features.

## Automated tests

The final command executed was:

```bat
.venv\Scripts\python.exe -m unittest checks test_gestures test_gesture_language test_language_integration test_gesture_interface test_launcher_ui test_interface_engine -v
```

**82 tests passed; 0 failures, 0 errors, 0 skipped.** All automated application launches were mocked. Coverage includes:

- Existing face models, owner decisions, enrollment, face IDs, hand tracking and v0.1 sequencing.
- Menu activation, all four selections and wrapping, exact destinations, confirmation and cancellation.
- Inactivity, held gestures, neutral release, classification dropouts, cooldown and fresh activation.
- Hand disappearance, changed hand/epoch, ambiguity, reordered hands, stale/cached samples and timestamp gaps.
- Exactly one launch per confirmation, invalid indices/URLs, slow browser startup and launch failure.
- Real Tk creation, initially hidden menu, four labels, cyan selection, cancellation, G, native close and main-thread enforcement.
- GUI callbacks during a deliberately slow camera worker, visible camera errors and manual exit stopping the worker.
- A single camera instance, continued headless frame processing, no OpenCV GUI calls in the worker, and no legacy sequence instantiation.

`pip check`, `START.bat --help` and `git diff --check` also passed. Installed runtime: Windows 10 build 19045, Python 3.11.9 x64, Tk/Tcl 8.6/8.6.12, NumPy 2.4.6, opencv-python 4.14.0.94 and MediaPipe 0.10.31. No dependency installation was needed for the extension.

## Real-camera usability findings

The user performed several camera sessions. The first run exposed the repeated-Victory issue; it was corrected with the neutral-release mechanism and regression tests. In the subsequent 90-second session, the logs recorded six menu activations, nine selection changes, one cancellation and five launch confirmations: Browser twice and ChatGPT, YouTube and GitHub once each. Face recognition continued in the background. No GUI callback error occurred.

That prototype still used a blank-tab Browser destination. Windows displayed an unsupported about-protocol chooser despite its launch API returning success. A direct-default-browser workaround was tested, producing a real blank Chrome tab. The user then requested Google instead. The final implementation uses the fixed Google HTTPS URL without protocol/registry workarounds. A separate real call through the final asynchronous launcher returned one successful Browser launch, and automated tests verify its exact Google URL. The final Google mapping and English text were not followed by another full user-camera session; the menu controls themselves were exercised in the earlier sessions.

The real-camera sessions reported GUI poll p95 values around 21 ms and orderly camera-worker shutdown. Manual exit was exercised by automated real-Tk tests; the timed user sessions shown in the saved logs ended on their time limit. These distinctions are retained rather than treating an API return as proof of successful page rendering or every usability criterion.

## Controlled comparison against v0.1

The benchmark uses the archived exact source of commit `1b3bd81` and the final v0.2 runtime. Its input is the public [MediaPipe woman_hands.jpg fixture](https://storage.googleapis.com/mediapipe-assets/woman_hands.jpg), cropped to `(0, 250, 640, 480)`. Both versions receive identical pixels, the same public models, the same read-only local reference, six OpenCV threads and a 20-Hz hand-inference ceiling. Real face/hand inference and real GUI event loops are included. Application launching is mocked.

Each run has 3 s warmup and 15 s measurement. Order is v0.1 / hidden menu / visible menu / visible menu / hidden menu / v0.1. The visible condition forces only the menu renderer open, without synthesizing a gesture or action. This is **unpaced processing throughput**, not sensor-limited webcam FPS. v0.1 includes its original OpenCV camera rendering; v0.2 uses its normal background engine and Tk indicator/menu.

| Condition | Run 1 FPS | Run 2 FPS | Median FPS |
|---|---:|---:|---:|
| Gesture Language v0.1 | 24.379229 | 16.576282 | 20.477755 |
| v0.2 menu hidden | 29.605020 | 29.488869 | 29.546944 |
| v0.2 menu visible | 30.066858 | 29.549258 | 29.808058 |

The median changes are **+44.29% hidden** and **+45.56% visible**, so neither v0.2 condition shows degradation and both meet the less-than-5% degradation target. The v0.1 runs vary substantially. The percentage should not be interpreted as a guaranteed general speedup; even the slowest measured v0.2 run exceeds the fastest measured v0.1 run in this test. Removing per-frame developer rendering is an architectural workload difference, not an improvement to the AI models.

All measured frames contained one face and two detected hands: 615 frames for v0.1, 888 hidden-menu frames and 896 visible-menu frames. Tk poll p95 was approximately 34.8–34.9 ms under this unpaced full-inference load. There were no recorded callback errors, no actual browser launches and all workers shut down. The full per-run data, source hashes, fixture hash and timings are in the local measurement file; see the [public benchmark summary](../benchmarks/gesture_interface_v0.2_summary.json). Validation confirms that the measured runtime hashes match the delivered code.

## Limitations and operation

This remains an experiment on one computer and a small number of user sessions. Gesture accuracy depends on pose, lighting and camera visibility. Geometric tracking cannot guarantee physical identity through every occlusion or very fast crossing. A confident but incorrect neutral classification could release a latch; a fresh confirmed command is still required before an action. Processing-throughput numbers do not include camera exposure, driver buffering or network page loading.

One GUI root owns all UI calls. Exit sets a stop event and closes Tk immediately; the camera worker is joined for at most two seconds. Normal shutdown releases the existing capture in its finally block. A driver that indefinitely blocks in native capture code may outlast that join; the recorded worker-stopped flag exposes that condition. It was not observed in these tests.

Run `START.bat` for the launcher and `START.bat --developer` for camera diagnostics/enrollment. The original v0.1 branch and backup remain available for reversal. Current release instructions are in [README.md](../README.md). No biometric profile, private capture or local experimental data was committed, and nothing was pushed to GitHub.

## Official references checked

- [Python 3.11 webbrowser](https://docs.python.org/3.11/library/webbrowser.html): fixed URLs can use the default browser with a requested new tab.
- [Python 3.11 Tkinter threading model](https://docs.python.org/3.11/library/tkinter.html#threading-model): event handlers must remain short; long-running work can run in another thread. The implementation keeps Tk calls on its owning main thread.
