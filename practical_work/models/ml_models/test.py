from gradio_client import Client

client = Client("https://sergiu2404-fin-tinybert-space.hf.space", hf_token="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", serialize=False)

result = client.predict("The company just announced profit exceeds expectation for this year.")
print(result)

