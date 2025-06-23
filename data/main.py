import pandas as pd
import torch
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.pipelines import pipeline as transformers_pipeline

OLLAMA_MODEL_NAME = "hf.co/LVSTCK/domestic-yak-8B-instruct-GGUF:Q8_0"
OLLAMA_HOST = "https://llama3.finki.ukim.mk"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = "gpt-4.1-mini"

MKLLM_HF_MODEL_ID = "trajkovnikola/MKLLM-7B-Instruct"


df = pd.read_csv("../input.csv", sep="\t")
df.rename(
    columns={
        "Question": "question",
        "Document": "document",
        "Answer": "ground_truth_answer",
    },
    inplace=True,
)
df["ground_truth_answer"] = df.apply(
    lambda row: row["document"]
    if row["ground_truth_answer"] == "-"
    else row["ground_truth_answer"],
    axis=1,
)


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Вие сте корисен асистент. Одговорете исклучиво врз основа на дадениот контекст.
Одговорот мора да биде точен и концизен извадок од контекстот. Не додавајте дополнителни информации или објаснувања.
Ако одговорот не може да се пронајде во контекстот, одговорете со: "Нема одговор".

Контекст:
{document}""",
        ),
        ("user", "Прашање: {question}\nОдговор:"),
    ]
)


def get_llm_response(llm_client_instance, question: str, document: str) -> str:
    formatted_prompt = prompt_template.invoke(
        {"question": question, "document": document}
    )
    ai_message_response = llm_client_instance.invoke(formatted_prompt)
    response_text = ai_message_response.content

    if "<|im_end|>" in response_text:
        response_text = response_text.split("<|im_end|>", 1)[0].strip()
    if response_text.endswith("."):
        response_text = response_text.rstrip(".")

    answer_prefix = "Одговор:"
    if answer_response := response_text.split(answer_prefix, 1):
        if len(answer_response) > 1:
            extracted_answer = answer_response[1].strip()
        else:
            extracted_answer = response_text.strip()
    else:
        extracted_answer = response_text.strip()

    if "Контекст:" in extracted_answer and extracted_answer.index("Контекст:") > 0:
        extracted_answer = extracted_answer.split("Контекст:", 1)[0].strip()
    if "Прашање:" in extracted_answer and extracted_answer.index("Прашање:") > 0:
        extracted_answer = extracted_answer.split("Прашање:", 1)[0].strip()

    return extracted_answer.split("\n")[0].strip()


ollama_llm_client = ChatOllama(
    model=OLLAMA_MODEL_NAME,
    base_url=OLLAMA_HOST,
    temperature=0,
    stop=["\n\n", "Context:", "Question:", "Контекст:", "Прашање:"],
    num_predict=200,
)

openai_llm_client = ChatOpenAI(
    model=OPENAI_MODEL_NAME,
    temperature=0,
    max_tokens=200,
    api_key=OPENAI_API_KEY,
)

mkllm_llm_client = None
try:
    local_tokenizer = AutoTokenizer.from_pretrained(MKLLM_HF_MODEL_ID)
    local_model = AutoModelForCausalLM.from_pretrained(
        MKLLM_HF_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )

    hf_pipeline = transformers_pipeline(
        "text-generation",
        model=local_model,
        tokenizer=local_tokenizer,
        max_new_tokens=200,
        do_sample=False,
        return_full_text=False,
        eos_token_id=local_tokenizer.eos_token_id,
    )

    mkllm_llm_client = ChatHuggingFace(
        llm=HuggingFacePipeline(pipeline=hf_pipeline),
        temperature=0,
    )
except Exception as e:
    print(
        f"ERROR: Could not load or initialize MKLLM-7B-Instruct from Hugging Face Hub: {e}"
    )


MODELS_TO_TEST = [
    {
        "llm_client": ollama_llm_client,
        "display_name": "Domestic",
        "model_id": OLLAMA_MODEL_NAME,
        "column_suffix": "_ollama_prediction",
    },
    {
        "llm_client": openai_llm_client,
        "display_name": "GPT 4.1 Nano",
        "model_id": OPENAI_MODEL_NAME,
        "column_suffix": "_openai_prediction",
    },
]

if mkllm_llm_client is not None:
    MODELS_TO_TEST.append(
        {
            "llm_client": mkllm_llm_client,
            "display_name": "MKLLM",
            "model_id": MKLLM_HF_MODEL_ID,
            "column_suffix": "_mkllm_prediction",
        }
    )


def collect_responses_to_dataframe():
    num_rows = len(df)

    for model_config in MODELS_TO_TEST:
        col_name = f"predicted_answer{model_config['column_suffix']}"
        df[col_name] = None

    for row_idx, row in df.iterrows():
        question = row["question"]
        document = row["document"]
        ground_truth_answer = row["ground_truth_answer"]

        print(f"\nProcessing row {row_idx + 1}/{num_rows} (Q: {question[:70]}...)")
        print(f"  Doc: {document[:70]}...")
        print(f"  GT: {ground_truth_answer}")

        for model_config in MODELS_TO_TEST:
            llm_client_instance = model_config["llm_client"]
            display_name = model_config["display_name"]
            column_suffix = model_config["column_suffix"]

            predicted_answer = ""
            if llm_client_instance is None:
                predicted_answer = "ERROR: Model not loaded/initialized."
            else:
                predicted_answer = get_llm_response(
                    llm_client_instance=llm_client_instance,
                    question=question,
                    document=document,
                )

            df.at[row_idx, f"predicted_answer{column_suffix}"] = predicted_answer

            print(f"       Pred ({display_name}): {predicted_answer[:70]}...")
        print("-" * 80)

    output_filename = "../output.csv"
    df.to_csv(output_filename, index=False, sep="\t")


if __name__ == "__main__":
    collect_responses_to_dataframe()
