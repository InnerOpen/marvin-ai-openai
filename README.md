# marvin-ai-openai

OpenAI for [Marvin](https://github.com/InnerOpen/marvin), as AI provider plugins. One package, two
providers:

- **`openai`**: the OpenAI API through the **Responses API** (every model takes the same length limit,
  function tools work while a model reasons, nothing is stored at OpenAI: `store=False`). With a custom
  base URL it talks to an **OpenAI-compatible server** (vLLM, LM Studio, a proxy) through Chat
  Completions instead, since most of them don't implement Responses.
- **`azure`**: **Azure OpenAI** deployments, through Chat Completions. It shares the request code with
  `openai`: a parameter a deployment refuses (`max_tokens` → `max_completion_tokens`, a temperature on a
  reasoning model) is corrected from its own 400 and remembered; a deployment that refuses tools while
  it reasons moves its tool calls to the Responses API (that needs an `api_version` that has it).

No model names are coded anywhere except the price table: a new model works the day it ships.

## Install

Marvin installs site-wide plugins at pod start from the chart's `plugins.packages` list. Only a
platform admin installs plugins; every workspace can then pick the provider in **Settings → AI**.

```yaml
plugins:
  packages:
    - https://github.com/InnerOpen/marvin-integration-sdk/archive/refs/heads/develop.tar.gz
    - https://github.com/InnerOpen/marvin-ai-openai/archive/refs/heads/main.tar.gz
```

Requires Marvin with the AI provider contract (`marvin-integration-sdk` 0.8+). Marvin core still has
built-in `openai` and `azure` providers; once this plugin is installed it **replaces** them (the log
says `AI provider plugin 'openai' replaces core's built-in 'openai'`) and reads the same settings, so
nothing else changes. **Admin → Plugins** lists `marvin-ai-openai` with kind *AI provider*.

> pip puts `openai` and its dependencies (`httpx`, `pydantic`, `pydantic-core`, `anyio`, `jiter`…)
> into the plugins volume, and `PYTHONPATH` comes before the image's site-packages, so those copies
> replace the image's own versions at runtime. Until the chart drops packages the image already has,
> check that the versions pip picks match Marvin's lock before installing this in production.

## Settings

Platform credentials (every workspace in *platform* credential mode):

| Provider | Setting | |
|---|---|---|
| openai | `OPENAI_API_KEY` | secret, masked |
| openai | `OPENAI_BASE_URL` | optional: an OpenAI-compatible server |
| azure | `AZURE_API_KEY` | secret, masked |
| azure | `AZURE_BASE_URL` | required: `https://<resource>.openai.azure.com` |
| azure | `AZURE_API_VERSION` | default `2024-02-01` |

A workspace in *workspace* credential mode uses its own key (a workspace secret) and, for Azure, a
provider row with the endpoint and `api_version` in its metadata.

## Prices

`prices.py` holds OpenAI's standard-tier prices per model (USD per million tokens). A dated snapshot
(`gpt-4o-2024-08-06`) takes its base id's price; a model that isn't listed has no estimate, and Marvin
shows the run as unpriced rather than free. A new price is a release of this package, not of Marvin.

## Develop

```sh
uv run --extra dev pytest -q
uv run --extra dev ruff check . && uv run --extra dev ruff format --check .
```

The tests never call OpenAI. `tests/fake_server.py` is a fake OpenAI server behind
`httpx.MockTransport`, so requests and answers still go through the `openai` SDK's own encoding and
parsing; the SDK's conformance kit (`marvin_integration_sdk.ai.testing`) runs over it for the official
API, an OpenAI-compatible server and Azure.
