from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
def interface():
    model = AutoPeftModelForCausalLM.from_pretrained("LLMLover/MODEL_EPOCHS_D2_testcase_2", torch_dtype=torch.float16, load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained("LLMLover/MODEL_EPOCHS_D2_testcase_2")
    new_model_name_D = "MODEL_D"
    model.save_pretrained(new_model_name_D)
    tokenizer.save_pretrained(new_model_name_D)

    content= '''

    import streamlit as st
    import textwrap
    import numpy as np
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer,BitsAndBytesConfig
    import torch
    import re

    models = {
        "Model_D": "LLMLover/MODEL_EPOCHS_D2_testcase_2"
    #    "Model_D": "Model_D"
    }

    def get_model():
        model = AutoPeftModelForCausalLM.from_pretrained("MODEL_D", torch_dtype=torch.float16, load_in_4bit=True)
        tokenizer = AutoTokenizer.from_pretrained("MODEL_D")
        return tokenizer, model

    def main():

        st.title("BOTH THE MODEL HAS SAME OR EVEN RESULT IN THE MY EVALUATION")
        st.title("EITHER SELECT MODEL B AND MODEL D")
        selected_model = st.selectbox("PLEASE SELECT A MODEL", list(models.keys()))
        model_name = models[selected_model]
        tokenizer, model = get_model()

        user_input = st.text_area("ENTER THE PYTHON QUESTION")
        button = st.button("Analyze")

        if user_input and button:
            Introduction_Sentence = "Below is an Python Programing question that describes a task. Write a Code for the same:"
            user_Input = "Input:"
            Response_Text = "### Response:"

            Introduction = f"{Introduction_Sentence}"
            input_context = f"{user_Input}\\n{user_input}"
            response = f"{Response_Text}\\n"

            parts = [part for part in [Introduction,input_context, response] if part]
            prompt= "\\n\\n".join(parts)
            input_ids = tokenizer(prompt, return_tensors='pt',padding=True,truncation=True)
            outputs = model.generate(**input_ids, max_new_tokens=256, temperature=0.7, top_p=0.6)

            output= tokenizer.decode(outputs[0][len(input_ids["input_ids"][0]):], skip_special_tokens=True)

            pattern = r'def\s+\w+\([^)]*\):\s*([\s\S]*?)\s*### End'
            pattern_2 = r'from\s+(\w+(\.\w+)*)\s*import\s+(\w+(,\s*\w+)*)\s*(as\s+\w+)?(?:,\s*(\w+(,\s*\w+)*)\s*(as\s+\w+)?)*'

            match = re.search(pattern, output.strip())
            match_2 = re.search(pattern_2, output.strip())

            if match:
                def_statement = match.group(0)
                if match_2:
                    extra_code=match_2.group(0)
                    combined_code = extra_code + "\\n" + def_statement
                    response=combined_code
                else:
                    response=def_statement

            else:
                match = re.search(r"def\s+(\w+)\s*\((.*)\):\s*return\s*(.*)", output.strip())
                if match:
                    def_statement = match.group(0)
                    if match_2:
                        extra_code=match_2.group(0)
                        combined_code = extra_code + "\\n" + def_statement
                        response=combined_code
                    else:
                        response=def_statement
                else:
                    response="INVALID CODE ERROR WITH THE VALUE "

            st.write(f"THIS IS THE PREDICTION: \\n {textwrap.dedent(response)}")

    if __name__ == "__main__":
        main()
    '''
    file_name="./custome_model_B_D.py"
    with open(file_name, 'w') as files:
        files.write(content)
    return file_name
    