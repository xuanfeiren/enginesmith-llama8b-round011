# llama8b-round011

A specialized inference engine for [NousResearch/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B-Instruct) at revision `d10aef7999a2b5ba950ab3974312feeedbfe0b77`, built with EngineSmith 0.3.0.dev0.

- Machine: 16 x Trainium (trn2.48xlarge), 64 cores, 24 GiB, Ubuntu 24.04.4 LTS
- Reference: nxdi 0.10.18399+ed62453e
- Judged under evaluation version 3
- Decode: one forward pass per generated token, no speculative decoding (a rule in the session brief; the harness cannot check it)
- Source and data: https://github.com/xuanfeiren/enginesmith-llama8b-round011
- Search: 11 rounds by Claude Code 2.1.317 (Claude Code) (model us.anthropic.claude-opus-5-5, reasoning effort medium), agent cost $1137.21

## Notes from the author

- The run's own final evaluation of round 011 did not start: a NeuronCore it needed was still held by another process, so the engine failed to load before judging anything. The numbers below are therefore the search evaluation's.
- A clean rebuild of round 011, checked afterwards on fresh held-out prompts (5 rounds, the engine on one core): passed; score 1.66x NxD Inference (prefill 2.01x, decode 1.37x). Decode probe 1.01.

## Speed against nxdi (search evaluation)

Score 1.66x; prefill 2.01x, decode 1.37x (reference time / engine time, 65 tokens per request; the fraction of the hardware bound is bound time / engine time, an estimate).

| prompt tokens | prefill | decode | prefill, fraction of the bound | decode, fraction of the bound |
|---:|---:|---:|---:|---:|
| 256 | 2.42x | 1.31x | not computed | not computed |
| 1,024 | 2.32x | 1.46x | not computed | not computed |
| 4,096 | 1.45x | 1.34x | not computed | not computed |

## Speed in absolute terms

Medians over the paired rounds of the search evaluation, this engine and nxdi on the same machine.

| prompt tokens | time to first token, ms | prefill tok/s | decode tok/s |
|---:|---:|---:|---:|
| 256 | 28.3 (nxdi 68.6) | 9,037.8 (3,730.1) | 44.7 (34.1) |
| 1,024 | 104.1 (nxdi 241.7) | 9,837.2 (4,236.7) | 44.6 (30.5) |
| 4,096 | 419.4 (nxdi 607.0) | 9,765.2 (6,747.8) | 42.6 (31.8) |

## Checks on the decode speed

- Natural text: not measured (no task accuracy answers).
- Decode probe: failed (see the run). One forward pass per generated token keeps this near 1; speculative decoding raises it.

## How the search went

- 11 coding sessions by Claude Code 2.1.317 (Claude Code) (model us.anthropic.claude-opus-5-5, reasoning effort medium); the search took 2543 min.
- First correct engine: round 001, 176 min after the start.
- First engine as fast as nxdi (score 1.0): round 001, 176 min after the start.
- Agent usage: 3,702 input and 2,045k output tokens, 453,605k cache reads, $1137.21 (as the agent reported them).

| round | kind | outcome | ended, min | score | prefill | decode | tokens in / out | engine lines |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 001 | generate | accepted (first_passing_candidate) | 175.9 | 1.37x | 1.54x | 1.23x | 308 / 223k | 1,972 |
| 002 | optimize | accepted (score_improved) | 328.2 | 1.49x | 1.74x | 1.28x | 214 / 96k | 8,423 |
| 003 | optimize | accepted (score_improved) | 588.0 | 1.54x | 1.82x | 1.30x | 408 / 195k | 8,720 |
| 004 | optimize | accepted (score_improved) | 848.0 | 1.57x | 1.90x | 1.29x | 274 / 188k | 9,339 |
| 005 | optimize | accepted (score_improved) | 1,126.7 | 1.59x | 1.90x | 1.33x | 604 / 196k | 9,675 |
| 006 | optimize | accepted (score_improved) | 1,276.9 | 1.62x | 1.91x | 1.38x | 130 / 41k | 9,748 |
| 007 | optimize | rejected (insufficient_improvement) | 1,469.8 | 1.62x | 1.91x | 1.37x | 242 / 174k | 9,909 |
| 008 | optimize | accepted (segment_improved) | 1,791.7 | 1.63x | 1.94x | 1.37x | 406 / 261k | 10,191 |
| 009 | optimize | accepted (segment_improved) | 2,028.4 | 1.64x | 1.98x | 1.37x | 326 / 215k | 10,355 |
| 010 | optimize | rejected (insufficient_improvement) | 2,175.2 | 1.65x | 1.98x | 1.37x | 332 / 168k | 10,433 |
| 011 | optimize | accepted (segment_improved) | 2,543.0 | 1.66x | 2.01x | 1.37x | 458 / 289k | 11,443 |

`history.csv` in https://github.com/xuanfeiren/enginesmith-llama8b-round011 has every round with every prompt length; `enginesmith.json` has the same and more.

## Use it

These are the author's measurements. Check them on a machine like this one (16 x Trainium (trn2.48xlarge); this builds the engine from `source/` and compares it with your local nxdi):

```sh
hf download NousResearch/Meta-Llama-3.1-8B-Instruct --revision d10aef7999a2b5ba950ab3974312feeedbfe0b77 --local-dir models/Llama-3.1-8B-Instruct
uv run enginesmith verify https://github.com/xuanfeiren/enginesmith-llama8b-round011 --model models/Llama-3.1-8B-Instruct
```

Build it yourself: `bash build.sh` in `source/`, then run `target/engine --model MODEL_DIR`; it speaks the engine protocol of EngineSmith's `docs/protocol.md` (token ids in, greedy token ids out). Read the source before you run it: it is native code written by a coding agent.

`enginesmith.json` holds every setting and result, and the SHA-256 of each model file.

## License

MIT, see `LICENSE`. Code adapted from other projects keeps its own license and notices (`source/THIRD_PARTY.md` lists it when the agent recorded any).
