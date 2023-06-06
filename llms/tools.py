import openai


class BaseTool:
    def __init__(self, model="gpt-3.5-turbo"):
        self.system = ""
        self.model = model
        self.message = [{"role": "system", "content": self.system}]

    def __call__(self, message):
        user_message = {"role": "user", "content": message}
        messages = self.message + [user_message]
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=messages
        )
        assistant_message = completion.choices[0].message
        return assistant_message["content"].replace("\n", " ")




class PreprocessingBot(BaseTool):
    def __init__(self, model="gpt-3.5-turbo"):
        super().__init__(model)
        self.system = r"""You are an AI assistant for raw data pre-processing. The user will input multiple raw references which may include unicode characters or ASCII code such as '\u001e'. Your task it to make it more readable by doing:
                      - Change all unicode characters or ASCII code such as '\u001e' to LaTeX format and put them in formula environment $...$ or $$...$$. 
                      - Re-write formulas or mathematical notations to LaTeX format in formula environment $...$ or $$...$$.
                      - Remove meaningless contents. 
                      - Response in the following format: {pdf-name-1: main contents from pdf-name-1, pdf-name-2: main contents from pdf-name-2, ...}. 
                      """
        self.message = [{"role": "system", "content": self.system}]

class ToolBot(BaseTool):
    def __init__(self, model="gpt-3.5-turbo"):
        super().__init__(model)
        self.system = r"""You need to pretend a Python function. You receive a string that is the user's question to a QA bot. You need to analyze the user's goal and decide if the QA bot needs to use the search engine to generate the response to the user. 
        Response 1 if you think the QA bot needs to use the search engine to user's input and response 0 if the QA bot doesn't need that.
                              """
        self.message = [{"role": "system", "content": self.system}]

if __name__ == "__main__":
    bot = ToolBot()
    rsp = bot("Hello!")
    print(rsp)
