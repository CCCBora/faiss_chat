import openai
import copy

class OpenAIChatBot:
    def __init__(self, model="gpt-3.5-turbo"):
        self.system = "You are Q&A bot. A highly intelligent system that answers user questions based on the information provided by the user's local database. " \
                      "User's question will include some references information above his question." \
                      "You need to answer user's question based on the provided references and inform the user what is the source of that reference. " \
                      "If you cannot find answer in the provided references, you still need to answer user's question but you also need to notice the user that your response is not based on the provided references."
        self.model = model
        self.message = [{"role": "system", "content": self.system}]
        self.raw_message = [{"role": "system", "content": self.system}]

    def load_message(self, message, role, original_message=None):
        if original_message is None:
            original_message = message
        msg = {"role": role, "content": message}
        self.message.append(msg)
        msg = {"role": role, "content": original_message}
        self.raw_message.append(msg)

    def load_chat(self, chat):
        msg = {"role": "user", "content": chat[0]}
        self.message.append(msg)
        msg = {"role": "assistant", "content": chat[1]}
        self.raw_message.append(msg)


    def __call__(self, message, original_message = None):
        self.load_message(message, "user", original_message)
        augmented_message = copy.deepcopy(self.message)
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=augmented_message
        )
        assistant_message = completion.choices[0].message
        return assistant_message["content"]
