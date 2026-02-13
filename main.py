# smoke.py
import mlx.core as mx
from mlx_lm import load, generate as mlx_generate
import sys

# Flush prints immediately
import functools
print = functools.partial(print, flush=True)

from src.llama.generate import prefill_system_prompt
from src.llama.patch import patch_model_attention
from src.cache import TriCache, CacheConfig

SYS_PROMPT = "You are a helpful AI assistant."


def main():
    print("Loading model...")
    model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct")

    print("Patching model with TriCache...")
    patch_model_attention(model)

    print("Prefilling system prompt...")
    system_prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYS_PROMPT}],
        add_generation_prompt=False,
        add_special_tokens=True,
        tokenize=True
    )
    prefill_system_prompt(model, mx.array(system_prompt_tokens))

    print("System prompt prefilled successfully.")

    # Simple test prompts
    test_prompts = [
        "How many 's's are in 'Mississippi'?",
        "What is the capital of France?",
        "Explain quantum computing in one sentence."
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}: Prompt = '{prompt}' ---")

        # Prepare user message without system prompt (since it's prefilled)
        messages = [{"role": "user", "content": prompt}]
        input_tokens = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            add_special_tokens=False,  # No BOS since system prompt already has it
            tokenize=True  # Returns List[int]
        )

        # Convert to string prompt for mlx_generate (or use tokens)
        prompt_str = tokenizer.decode(input_tokens)
        print(f"[DEBUG] Prompt string passed to generate: {prompt_str}")

        try:
            print("[DEBUG] Calling mlx_generate...")
            response = mlx_generate(
                model,
                tokenizer,
                prompt=input_tokens,
                max_tokens=10,
                verbose=True,
            )

            print(f"Generated response: {response}")
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    print("\nSmoke test completed successfully! No major crashes detected.")


if __name__ == "__main__":
    main()