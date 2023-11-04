import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import HuggingFaceHub
from getpass import getpass
import os
from dotenv import load_dotenv

load_dotenv()

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define the HuggingFace model repository ID
repo_id = "Salesforce/codegen2-1B"
# repo_id = "SEBIS/code_trans_t5_small_code_documentation_generation_python"

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 1, "max_length": 200})

# Streamlit app
st.title("Code Migration App")

# Input source language
source_language = st.text_input("Source Language", "python")

# Input target language
target_language = st.text_input("Target Language", "c++")

# Input source code
source_code = st.text_area("Source Code", """
for i in range(12):
    print(i)
""")

# Button to trigger the migration
if st.button("Migrate Code"):
    # Define the PromptTemplates for each step in the migration process
    parsing_template = """
    analyse this {source_language} code: 
    {source_code}
    """

    generation_template = """
    generate {target_language} code from the parsed {source_language} code using this information {parsed_info}
    """

    parsing_prompt_template = PromptTemplate(
        input_variables=["source_language", "source_code"], template=parsing_template
    )
    parsing_chain = LLMChain(
        llm=llm, prompt=parsing_prompt_template, output_key="parsed_info"
    )

    generation_prompt_template = PromptTemplate(
        input_variables=["target_language", "source_language", "parsed_info"],
        template=generation_template
    )

    generation_chain = LLMChain(
        llm=llm, prompt=generation_prompt_template, output_key="generated_code"
    )

    overall_chain = SequentialChain(
        chains=[parsing_chain, generation_chain],
        input_variables=["source_language", "target_language", "source_code"],
        output_variables=["generated_code"]
    )

    target_code = overall_chain({
        "source_language": source_language,
        "target_language": target_language,
        "source_code": source_code
    })

    # Display the generated code
    st.subheader("Generated Code")
    st.code(target_code['generated_code'], language=target_language)
