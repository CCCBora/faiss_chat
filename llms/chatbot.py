import openai

class OpenAIChatBot:
    def __init__(self, model="gpt-3.5-turbo"):
        self.system = "You are Q&A bot. A highly intelligent system that answers user questions based on the information provided by the user's local database. " \
                      "If the user has connected to the local database, his question will include some references information above his question." \
                      "If the user has not connected to the local database, you cannot access the references information. " \
                      "You need to answer user's question based on the provided references and inform the user the name of PDF file. " \
                      "If you cannot find related references, you still need to answer user's question but you also need to notice the user that your response is not based on the provided references."
        self.model = model
        self.message = [{"role": "system", "content": self.system}]

    def __call__(self, message):
        user_message = {"role": "user", "content": message}
        self.message.append(user_message)

        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=self.message
        )
        assistant_message = completion.choices[0].message
        self.message.append(assistant_message)
        return assistant_message["content"]