# freeroute-aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/joker31313131/freeroute-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/joker31313131/freeroute-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

FreeRoute is a single server that serves the OpenAI Chat Completions API and the Anthropic Messages API, answering each prompt with a real completion from the best-ranked live free OpenRouter model, dialled through the grader's gateway, with failover.

## Run it

```sh
node main.ts            # trial mode: reads MAKEFUN_PROMPT_FILE, registers, serves, polls /work
node main.ts --list     # print the catalog document it registers, byte for byte, and exit 0
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=h7j0gh6r&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
