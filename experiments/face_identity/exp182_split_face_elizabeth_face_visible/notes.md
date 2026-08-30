---
status: ready
concept: face
method: frame_replace_split/precompute
thread: face_identity
takeaway: >
  Prompt-focused follow-up to exp180. Its close-up prompts often make hands or handled objects the
  active visual subject, so this run tests 30 static head-and-shoulders portrait triples that keep
  Queen Elizabeth II's unobstructed frontal face centered throughout and use only facial motion.
  All sampler settings are held fixed and both prediction/trajectory modes are retained. Not yet
  submitted.
---
# exp182 — face-visible Queen Elizabeth II split-prompt dataset

## Why

Human inspection of exp180 found that many concept clips do not actually show Queen Elizabeth II's
face and instead become close-ups of hands or objects. The word *close-up* was not enough to control
the subject of the crop: exp180 frequently pairs it with actions such as signing letters, inspecting
seedlings, handling a vase, plucking a harp, or arranging objects. CogVideoX can satisfy those
prompts by making the action and the manipulated object dominate the frame.

This follow-up changes prompt semantics, not sampler settings. Every A prompt explicitly asks for:

- one person in a static head-and-shoulders or shoulders-up portrait;
- Queen Elizabeth II's full, unobstructed, front-facing face as the large central subject;
- direct eye contact with the lens;
- only low-amplitude facial motion (speaking, smiling, blinking, or one small nod);
- a fixed camera and softly blurred or plain background.

There are no hand actions or foreground props. Clothing, room, and lighting vary enough to avoid
turning one portrait costume into the learned concept. B mirrors A's composition and facial motion
with a visually distinct anonymous woman; C retains a generic woman and the same face-dominant
composition so the shared heal phase does not remove the face.

## Setup

`prompts/face_identities/split/queen_elizabeth_ii_face_visible.csv` contains 30 new A/B/C triples
with seeds 8001–8030. These seeds are disjoint from exp180 (7901–7990) and the earlier face datasets.
The config matches exp180 on inference, split, detector, diagnostic, and time-limit fields. It grids
`split_mode` over `prediction` and `trajectory`, producing two jobs; this keeps the prompt comparison
usable even before exp180's winning sampler arm is known.

The new set is intentionally 30 rows rather than another 90. This is a targeted prompt-design test:
if face visibility does not improve under such strong composition control, spending another 4x the
compute on equivalent prompts is not justified.

## Pre-registered checks

- **Face rendering:** whole-clip A videos should contain a detectable face in at least 24/30 rows
  (80%) in either sampler arm. Review the videos as well as detector scores because no-face frames
  are encoded as confidence 0.
- **Identity rendering:** at least 15/30 whole-clip A rows should reach Queen Elizabeth II identity
  confidence 0.23, the calibrated identity threshold from exp090.
- **Failure-mode reduction:** no more than 3/30 A videos should be dominated by hands, a handled
  object, or a crop that excludes most of the face on human review.
- **Usable yield:** at least 15/30 rows should pass splice/identity-separation review in one sampler
  arm. If this passes but is too small for training, author two more seed-disjoint CSVs in the same
  portrait register rather than broadening back to prop-driven actions.
- **Balance:** retained concept regions should remain roughly balanced between first and second.

Compare whole-clip face presence and identity confidence against each exp180 arm before comparing
final split yield. The prompt hypothesis is supported if face presence rises even when identity
confidence or split yield does not; those later failures would point to identity rendering or the
splitter rather than the crop-selection problem this experiment isolates.

## Status

- [x] Authored 30 face-dominant prompt triples with seeds 8001–8030.
- [ ] Submitted (project owners submit cluster jobs).
- [ ] Built and pulled.
- [ ] Screened with `tools/screen_split_dataset.py` and the face continuity screen.
- [ ] Whole-clip and edited videos reviewed for face visibility and identity separation.
- [ ] Gates evaluated and exp096 dataset choice updated.
