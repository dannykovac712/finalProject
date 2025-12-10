from openai import OpenAI
client = OpenAI()

resp = client.responses.create(
    model="gpt-4o-mini",
    input="Say hello in one word."
)
print(resp.output_text)
