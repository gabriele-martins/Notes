import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm  # Importa tqdm para a barra de progresso

load_dotenv()

client = OpenAI(
    api_key=os.getenv('LLAMA3_API_KEY'),
    base_url=os.getenv('LLAMA3_API_URL'),
)

def make_llama3_request(instruction, max_tokens=2048):
    response = client.chat.completions.create(
        model="llama-13b-chat",
        messages=[
            {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
            {"role": "user", "content": instruction},
        ],
        max_tokens=50
    )

    return response.choices[0].message.content.strip()

def split_text(text, max_length):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def reduce_text_for_title(text, max_words):
    words = text.split()
    return ' '.join(words[:max_words])

def generate_title(text):
    max_words = 50
    reduced_text = reduce_text_for_title(text, max_words)

    instruction = f"""
        A partir do seguinte texto reduzido, gere um título conciso e informativo que capture o tema principal:
        {reduced_text}

        Retorne apenas o título, sem aspas ou marcas adicionais.
    """

    return make_llama3_request(instruction, max_words)

def generate_initial_summary(text):
    instruction = f"""
        Resuma o seguinte texto em tópicos, focando apenas no conteúdo relevante e ignorando qualquer assunto irrelevante:
        {text}

        Produza um resumo categorizado que inclua:
        1. Um título principal.
        2. Introdução.
        3. Divisões de tópicos com subtítulos.
    """

    return make_llama3_request(instruction)

def generate_partial_summary(text):
    instruction = f"""
        Resuma o seguinte texto em tópicos, focando apenas nas partes relevantes e ignorando informações não essenciais:
        {text}

        Produza um resumo categorizado que inclua:
        Divisões de tópicos com subtítulos.
    """

    return make_llama3_request(instruction)

def generate_conclusion(text):
    instruction = f"""
        A partir do texto fornecido, gere uma conclusão final que resuma os principais pontos discutidos e ignore detalhes irrelevantes:
        {text}

        Produza uma conclusão que sintetize as informações principais.
    """

    return make_llama3_request(instruction, 512)

def generate_report_summary(text):
    text_chunks = split_text(text, 100)
    
    summaries = []
    for chunk in tqdm(text_chunks, total=len(text_chunks), desc='Processando partes'):
        partial_summary = generate_partial_summary(chunk)
        summaries.append(partial_summary)

        time.sleep(2)

    combined_summary = "\n\n".join(summaries)

    return combined_summary

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def save_to_markdown(text, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text)
    
    print(f'Arquivo salvo em {output_file}')

def bootstrap():
    file_path = "data/resumo.txt"
    text = read_text_from_file(file_path)
    print("Arquivo de texto carregado")

    title_name = generate_title(text)
    print(f'Título gerado para o texto: {title_name}')
    output_path = f"summaries/{title_name}.md"
    summary = generate_report_summary(text)
    save_to_markdown(summary, output_path)

if __name__ == "__main__":
    bootstrap()
