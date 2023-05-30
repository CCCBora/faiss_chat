import openai

class PreprocessingBot:
    def __init__(self, model="gpt-3.5-turbo"):
        self.system = r"""You are an AI assistant for raw data pre-processing. The user will input multiple raw references which may include unicode characters or ASCII code such as '\u001e'. Your task it to make it more readable by doing:
                      - Change all unicode characters or ASCII code such as '\u001e' to LaTeX format and put them in formula environment $...$ or $$...$$. 
                      - Re-write formulas or mathematical notations to LaTeX format in formula environment $...$ or $$...$$.
                      - Remove meaningless contents. 
                      - Response in the following format: {pdf-name-1: main contents from pdf-name-1, pdf-name-2: main contents from pdf-name-2, ...}. 
                      """
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