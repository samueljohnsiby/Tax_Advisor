from openai import OpenAI


class TaxAdvisor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def chat_with_user(self, messages):

        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages
        )
        response = completion.choices[0].message.content
        # self.messages.append({"role": "assistant", "content": response})
        return response
