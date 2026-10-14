"""What OpenAI charges, per model: USD per million tokens, standard tier.

From developers.openai.com/api/docs/pricing (2026-10-01). A dated snapshot (``gpt-4o-2024-08-06``)
takes the price of the id it extends; a model that isn't listed has no estimate (Marvin shows "—" and
counts the run as unpriced). A new model's price is a release of this package, not of Marvin.
"""

from marvin_integration_sdk.ai import ModelPrice


def _price(input_per_1m: float, output_per_1m: float) -> ModelPrice:
    """A model's price with OpenAI's cache rate: a prompt token its automatic cache served is a tenth of the input
    rate (``cache_read_input_token_cost`` in LiteLLM's feed, 2026-10-14); there is no write premium."""
    return ModelPrice(input_per_1m=input_per_1m, output_per_1m=output_per_1m, cached_input_per_1m=round(input_per_1m * 0.1, 6))


OPENAI_PRICES: dict[str, ModelPrice] = {
    # GPT-5 family — standard tier, from developers.openai.com/api/docs/pricing (2026-10-01).
    "gpt-5.6-sol": _price(4.00, 20.00),
    "gpt-5.6-terra": _price(2.00, 12.00),
    "gpt-5.6-luna": _price(0.20, 1.20),
    "gpt-5.5": _price(5.00, 30.00),
    "gpt-5.4": _price(2.50, 15.00),
    "gpt-5.4-mini": _price(0.75, 4.50),
    "gpt-5.4-nano": _price(0.20, 1.25),
    "gpt-5.2": _price(1.75, 14.00),
    "gpt-5.1": _price(1.25, 10.00),
    "gpt-5": _price(1.25, 10.00),
    "gpt-5-mini": _price(0.25, 2.00),
    "gpt-5-nano": _price(0.05, 0.40),
    "gpt-4o": _price(2.50, 10.00),
    "gpt-4o-mini": _price(0.15, 0.60),
    "gpt-4.1": _price(2.00, 8.00),
    "gpt-4.1-mini": _price(0.40, 1.60),
    "gpt-4-turbo": _price(10.00, 30.00),
    "o3": _price(10.00, 40.00),
    "o4-mini": _price(1.10, 4.40),
    # Image models — priced by the token usage the images API returns (input text, output image).
    "gpt-image-1": _price(5.00, 40.00),
    "gpt-image-1-mini": _price(2.00, 8.00),
    # The current generation (pricing page, 2026-10-14): text input $5, image output $30–32 per 1M tokens.
    "gpt-image-1.5": _price(5.00, 32.00),
    "gpt-image-2": _price(5.00, 30.00),
    "gpt-image-2.5-flare": _price(5.00, 30.00),
    "gpt-image-2.5-sunburst": _price(5.00, 30.00),
}

AZURE_PRICES: dict[str, ModelPrice] = {
    # Azure's list prices mirror OpenAI's for these deployments.
    "gpt-4o": OPENAI_PRICES["gpt-4o"],
    "gpt-4.1": OPENAI_PRICES["gpt-4.1"],
}
