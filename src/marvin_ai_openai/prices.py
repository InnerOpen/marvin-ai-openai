"""What OpenAI charges, per model: USD per million tokens, standard tier.

From developers.openai.com/api/docs/pricing (2026-10-01). A dated snapshot (``gpt-4o-2024-08-06``)
takes the price of the id it extends; a model that isn't listed has no estimate (Marvin shows "—" and
counts the run as unpriced). A new model's price is a release of this package, not of Marvin.
"""

from marvin_integration_sdk.ai import ModelPrice

OPENAI_PRICES: dict[str, ModelPrice] = {
    # GPT-5 family — standard tier, from developers.openai.com/api/docs/pricing (2026-10-01).
    "gpt-5.6-sol": ModelPrice(input_per_1m=4.00, output_per_1m=20.00),
    "gpt-5.6-terra": ModelPrice(input_per_1m=2.00, output_per_1m=12.00),
    "gpt-5.6-luna": ModelPrice(input_per_1m=0.20, output_per_1m=1.20),
    "gpt-5.5": ModelPrice(input_per_1m=5.00, output_per_1m=30.00),
    "gpt-5.4": ModelPrice(input_per_1m=2.50, output_per_1m=15.00),
    "gpt-5.4-mini": ModelPrice(input_per_1m=0.75, output_per_1m=4.50),
    "gpt-5.4-nano": ModelPrice(input_per_1m=0.20, output_per_1m=1.25),
    "gpt-5.2": ModelPrice(input_per_1m=1.75, output_per_1m=14.00),
    "gpt-5.1": ModelPrice(input_per_1m=1.25, output_per_1m=10.00),
    "gpt-5": ModelPrice(input_per_1m=1.25, output_per_1m=10.00),
    "gpt-5-mini": ModelPrice(input_per_1m=0.25, output_per_1m=2.00),
    "gpt-5-nano": ModelPrice(input_per_1m=0.05, output_per_1m=0.40),
    "gpt-4o": ModelPrice(input_per_1m=2.50, output_per_1m=10.00),
    "gpt-4o-mini": ModelPrice(input_per_1m=0.15, output_per_1m=0.60),
    "gpt-4.1": ModelPrice(input_per_1m=2.00, output_per_1m=8.00),
    "gpt-4.1-mini": ModelPrice(input_per_1m=0.40, output_per_1m=1.60),
    "gpt-4-turbo": ModelPrice(input_per_1m=10.00, output_per_1m=30.00),
    "o3": ModelPrice(input_per_1m=10.00, output_per_1m=40.00),
    "o4-mini": ModelPrice(input_per_1m=1.10, output_per_1m=4.40),
    # Image models — priced by the token usage the images API returns (input text, output image).
    "gpt-image-1": ModelPrice(input_per_1m=5.00, output_per_1m=40.00),
    "gpt-image-1-mini": ModelPrice(input_per_1m=2.00, output_per_1m=8.00),
}

AZURE_PRICES: dict[str, ModelPrice] = {
    # Azure's list prices mirror OpenAI's for these deployments.
    "gpt-4o": OPENAI_PRICES["gpt-4o"],
    "gpt-4.1": OPENAI_PRICES["gpt-4.1"],
}
