import re
import torch
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

stop_words_pt = [
    "a", "ao", "aos", "a", "com", "da", "das", "de", "do", "dos", "em", "para", "per", "por", "que", "um", "uma", "uns", "umas"
]

def test_gpu():
    print("GPU disponível:", torch.cuda.is_available())
    x = torch.tensor([1.0, 2.0, 3.0])
    print('Tensor simples para teste', x)

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def preprocess_text(text):
    # Remove URLs, e-mails e outras informações irrelevantes
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text, flags=re.MULTILINE)

    # Remove palavras e frases que não estão relacionadas
    text = re.sub(r'\b(oi|boa noite|gente|etc)\b', '', text, flags=re.IGNORECASE)

    # Remove múltiplos espaços e quebras de linha
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def split_into_paragraphs(text):
    # Divida o texto em parágrafos baseados em pontuações
    paragraphs = re.split(r'\.\s+', text)
    return [p.strip() for p in paragraphs if p.strip()]

def extract_topics(paragraphs, n_topics):
    print("Extraindo tópicos...")
    vectorizer = TfidfVectorizer(stop_words=stop_words_pt)
    X = vectorizer.fit_transform(paragraphs)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = []
    for topic_index, topic in tqdm(enumerate(lda.components_), desc="Extraindo tópicos", total=n_topics):
        top_keywords_index = topic.argsort()[-10:][::-1]
        top_keywords = [feature_names[i] for i in top_keywords_index]
        topic_keywords.append(' '.join(top_keywords))

    return topic_keywords, lda.transform(X)

def categorize_text(text):
    preprocessed_text = preprocess_text(text)
    paragraphs = split_into_paragraphs(preprocessed_text)

    print("Extraindo tópicos...")
    topics, topic_matrix = extract_topics(paragraphs, n_topics=5)  # Ajuste o número de tópicos conforme necessário

    print("Tópicos dos parágrafos extraídos")
    print(f"{len(paragraphs)} parágrafos detectados")

    categorized_text = {}
    for i, paragraph in enumerate(paragraphs):
        topic_index = np.argmax(topic_matrix[i])
        topic = topics[topic_index]

        if topic not in categorized_text:
            categorized_text[topic] = []
        categorized_text[topic].append(paragraph)
    
    return categorized_text

def capitalize_title(title):
    return title.capitalize()

def save_to_markdown(categorized_text, output_file):
    print(f"Salvando texto categorizado em {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as file:
        for topic, paragraphs in categorized_text.items():

            topic_capitalized = capitalize_title(topic)
            file.write(f"## {topic_capitalized}\n\n")
            for paragraph in paragraphs:
                file.write(f"{paragraph}\n\n")
            file.write("\n")

def bootstrap():
    test_gpu()

    file_path = "data/artigo-ia.txt"
    text = read_text_from_file(file_path)
    print("Arquivo de texto carregado")

    categorized_text = categorize_text(text)

    markdown_file = "texto_categorizado.md"
    save_to_markdown(categorized_text, markdown_file)
    print(f"Texto categorizado salvo em {markdown_file}")

if __name__ == "__main__":
    bootstrap()
