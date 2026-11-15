# Free Model Aggregator 02

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API
(JSON and streaming) and answers every prompt with a real completion from a live free
OpenRouter model, dialled through the grader's gateway on the entrant's pushed key.

Run it:

```sh
node main.ts            # trial mode: reads MAKEFUN_PROMPT_FILE, registers, serves, polls /work
node main.ts --list     # print the catalog document it registers, byte for byte, and exit 0
```

Node 22, standard library only. `main.ts` is the entrypoint; `lib/providers.ts` reads the
brief, ranks the free models by the house rating at each prompt's requirements, and builds
the catalog; `lib/wire.ts` translates Anthropic <-> Chat Completions and rewrites streamed
answers with the dialled model's id on every chunk.

Routing: best-rated eligible model first, failover down the ranking on refusals, per-model
cooldowns counted in prompts (never wall time), a per-prompt and per-trial dial cap, and an
`x-makefun-upstream-failures` trail on answers that recovered from a refusal. Every choice
derives from the brief, the requests, and the gateway's answers, so a replay reproduces the
same export byte for byte.

No key lives in this repository. The gateway swaps the trial's dummy key for the entrant's
own pushed OpenRouter key, which never enters the sandbox.
