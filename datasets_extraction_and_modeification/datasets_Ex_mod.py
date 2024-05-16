import os
from datasets import load_dataset
import requests
import json
import pandas as pd
import json
import time
import re
import textwrap
def Custom_Prompt_for_Code_Gen(sample):
    """
    The sample prompt will as flow :
    Below is an Python Programing question that describes a task. Write a Code for the same:

    Input: Write a python function to count numbers whose oth and nth bits are set.

    ### Response:
    def count_Num(n):
       if (n == 1):
       return 1
   count = pow(2,n - 2)
   return count

   ### End


    """

    Introduction_Sentence = "Below is an Python Programing question that describes a task. Write a Code for the same:"
    user_Input = "Input:"
    Response_Text = "### Response:"
    end_ouput = "### End"

    Introduction = f"{Introduction_Sentence}"
    input_context = f"{user_Input}\n{sample['text']}"
    response = f"{Response_Text}\n{sample['code']}"
    end = f"{end_ouput}"

    parts = [part for part in [Introduction,input_context, response, end] if part]

    sample['formatted_prompt'] = "\n\n".join(parts)
    #sample["combined_text"] = formatted_prompt
    return sample



def Loading_and_split(dataset_name):
  dataset = load_dataset(dataset_name, split="test").train_test_split(test_size=0.25)
  valid_train_dataset=dataset['train'].train_test_split(test_size=0.35)
  return valid_train_dataset,dataset['test']





def new_output(sample,model,tokenizer):
  Introduction_Sentence = "Below is an Python Programing question that describes a task. Write a Code for the same:"
  user_Input = "Input:"
  Response_Text = "### Response:"

  Introduction = f"{Introduction_Sentence}"
  input_context = f"{user_Input}\n{sample['text']}"
  response = f"{Response_Text}\n"

  parts = [part for part in [Introduction,input_context, response] if part]
  #print(parts)

  prompt= "\n\n".join(parts)
  input_ids = tokenizer(prompt, return_tensors='pt',padding=True,truncation=True)
  outputs = model.generate(**input_ids,
        max_new_tokens=256, temperature=0.7, top_p=0.6

    )
  output= tokenizer.decode(outputs[0][len(input_ids["input_ids"][0]):], skip_special_tokens=True)
  pattern = r'def\s+\w+\([^)]*\):\s*([\s\S]*?)\s*### End' ###r"def\s+(\w+)\s*\((.*)\):\s*return\s*(.*)"
  pattern_2 = r'from\s+(\w+(\.\w+)*)\s*import\s+(\w+(,\s*\w+)*)\s*(as\s+\w+)?(?:,\s*(\w+(,\s*\w+)*)\s*(as\s+\w+)?)*'

# Search for the def statement using the regular expression
  match = re.search(pattern, output.strip())
  match_2 = re.search(pattern_2, output.strip())

# Extract the def statement if a match is found
  if match:
    def_statement = match.group(0)
    if match_2:
        extra_code=match_2.group(0)
        combined_code = extra_code + "\n" + def_statement
        sample['output']=textwrap.dedent(combined_code)
    else:
        sample['output']=textwrap.dedent(def_statement)

  else:
    match = re.search(r"def\s+(\w+)\s*\((.*)\):\s*return\s*(.*)", output.strip())
    if match:
        def_statement = match.group(0)
        if match_2:
            extra_code=match_2.group(0)
            combined_code = extra_code + "\n" + def_statement
            sample['output']=textwrap.dedent(combined_code)
        else:
            sample['output']=textwrap.dedent(def_statement)
    else:
        sample['output']="INVALID CODE ERROR WITH THE VALUE "

  sample['unfilter']=output.strip()
  return sample





AWS_API_KEY = "38ED56FED62444439E90C2D855304844"
def llama_generate(prompt,
                   api_token,
                   max_gen_len = 512,
                   temperature = 0.5,
                   top_p =0.9):
  url = 'https://6xtdhvodk2.execute-api.us-west-2.amazonaws.com/dsa_llm/generate'
  body = {
    "prompt": prompt,
    "max_gen_len": max_gen_len,
    "temperature": temperature,
    "top_p": top_p,
    "api_token": api_token
  }
  res = requests.post(url,  json = body)
  return  json.loads(res.text)["body"]["generation"]


def genrate_code_from_LLM():
    data=[]
    i=0
    for j in range(1):
        time.sleep(20)
        for prompt in ["""
    Give me 6 python code using the following template for each Question.

    # Question: Write a function to find the similar elements from the given two tuple lists.

    # Code: def similar_elements(test_tup1, test_tup2): \n res = tuple(set(test_tup1) & set(test_tup2)) \n return (res)

    ####


    ""","""
    Give me 6 python code using the following template for each Question.

    # Question: Write a python function to identify non-prime numbers.

    # Code: import math def is_not_prime(n): result = False for i in range(2,int(math.sqrt(n)) + 1): if n % i == 0: result = True return result

    ####


    ""","""
    Give me 6 python code using the following template for each Question.

    # Question: Write a function to get the n smallest items from a dataset.

    # Code: import heapq def small_nnum(list1,n): smallest=heapq.nsmallest(n,list1) return smallest

    ####


    ""","""
    Give me 6 python code using the following template for each Question.

    # Question: Write a function to count the most common words in a dictionary.

    # Code: from collections import Counter def count_common(words): word_counts = Counter(words) top_four = word_counts.most_common(4) return (top_four)

    ####


    ""","""
    Give me 6 python code using the following template for each Question.

    # Question: Write a function to split the given string with multiple delimiters by using regex.

    # Code: import re def multiple_split(text): return (re.split('; |, |\*|\n',text))

    ####


    ""","""
    Give me 6 python code using the following template for each Question.

    # Question: Write a python function to multiply all items in the list.

    # Code: def multiply_list(items): tot = 1 for x in items: tot *= x return tot

    ####


    ""","""
    Give me 6 python code using the following template for each Question.

    # Question: Write a function to find the length of the shortest string that has both str1 and str2 as subsequences.

    # Code: def super_seq(X, Y, m, n): if (not m): return n if (not n): return m if (X[m - 1] == Y[n - 1]): return 1 + super_seq(X, Y, m - 1, n - 1) return 1 + min(super_seq(X, Y, m - 1, n), super_seq(X, Y, m, n - 1))

    ####


    ""","""
    Give me 6 python code using the following template for each Question.

    # Question: Write a function to find the maximum number of segments of lengths a, b and c that can be formed from n.

    # Code: def maximum_segments(n, a, b, c) : dp = [-1] * (n + 10) dp[0] = 0 for i in range(0, n) : if (dp[i] != -1) : if(i + a <= n ): dp[i + a] = max(dp[i] + 1, dp[i + a]) if(i + b <= n ): dp[i + b] = max(dp[i] + 1, dp[i + b]) if(i + c <= n ): dp[i + c] = max(dp[i] + 1, dp[i + c]) return dp[n]

    ####


    ""","""
    Give me 6 python code using the following template for each Question.

    # Question: Write a function to find the minimum total path sum in the given triangle.

    # Code: def min_sum_path(A): memo = [None] * len(A) n = len(A) - 1 for i in range(len(A[n])): memo[i] = A[n][i] for i in range(len(A) - 2, -1,-1): for j in range( len(A[i])): memo[j] = A[i][j] + min(memo[j], memo[j + 1]) return memo[0]
    ####


    """]:
            time.sleep(20)
            print(prompt)
            d=llama_generate(prompt, AWS_API_KEY)
            for sol in d.split("####"):
                try:
                    
                    if sol.split("# Question:")[1].split("\n\n")[0].strip()!='' and sol.split("# Code:")[1].strip()!='':
                        i=i+1
                    Question  = sol.split("# Question:")[1].split("\n\n")[0].strip()
                    Code=sol.split("# Code:")[1].strip()
                    data.append({'ID':i,'Text':Question,'Code':Code})
                except:
                    print("No value")
            # with open(f'Final_data{j}.json', 'w') as f:
        #     json.dump(data, f)

    with open('Final_data.json', 'w') as f:
        json.dump(data, f)
    return data