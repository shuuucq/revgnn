import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from data_preprocess.util_data_preprocess import read_json_file, read_list_from_txt

# Initialize OAG-BERT
local_model_path = "/home/aimingshu/project/Reviewer-Rec-main/sci-bert"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModel.from_pretrained(local_model_path)

def load_paper_id_text_from_file(paper_filepath, reviewer_filepath, paper_mapping_dict):
    id_text_dict = {}
    paper_detail = read_json_file(paper_filepath)
    reviewer_detail = read_json_file(reviewer_filepath)

    # Create a mapping for author names to their affiliations
    author_affiliation_map = {}
    for reviewer in reviewer_detail:
        name = reviewer['name']
        affiliations = [aff['Name'] for aff in reviewer.get('affiliation', [])]
        author_affiliation_map[name] = affiliations

    for each_paper in paper_detail:
        paper_id = each_paper['_id']  # Original paper ID
        mapped_id = paper_mapping_dict.get(paper_id, paper_id)  # If mapping exists, use the mapped value, otherwise keep the original ID
        
        id_text_dict[mapped_id] = {
            'title': each_paper['Title'],
            'abstract': each_paper.get('Abstract', ''),
            "keywords": each_paper.get('Keywords', []),
            "venue": each_paper.get('JournalTitle', ''),
            "authors": [],
            "affiliations": []
        }

        # Extract FullName for authors
        authors = [author['FullName'] for author in each_paper.get('Authors', [])]
        id_text_dict[mapped_id]["authors"] = authors

        # Look up each author's affiliations
        for author in authors:
            if author in author_affiliation_map:
                id_text_dict[mapped_id]["affiliations"].extend(author_affiliation_map[author])

    return id_text_dict


def get_sci_emb(input_dict):
    # Construct input text for OAG-BERT
    title = input_dict['title']
    abstract = input_dict.get('abstract', '')
    keywords = input_dict.get('keywords', [])
    if not isinstance(keywords, list):  # 确保 keywords 是一个列表
        keywords = []
    keywords_str = ', '.join(keywords)  # 将关键词连接成字符串
    venue = input_dict.get('venue', '')
    authors = ', '.join(input_dict.get('authors', []))
    affiliations = ', '.join(input_dict.get('affiliations', []))

    combined_text = f"{title} [SEP] {abstract} [SEP] {keywords_str} [SEP] {venue} [SEP] {authors} [SEP] {affiliations}"

    # 进行分词并创建输入张量，同时进行截断
    inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # 从 SciBERT 获取嵌入表示
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]  # 提取 [CLS] token 的嵌入
        # print(embedding.shape)

    return embedding.flatten().cpu().tolist()  # 将 Tensor 转换为 Python 列表


def get_paper_embedding(id_text_dict: dict):
    paper_embedding_dict = {}
    total_number = len(id_text_dict)
    count = 1
    for key, value in id_text_dict.items():

        embedding = get_sci_emb(value)
        if embedding is not None:
            # print(len(embedding))
            # print(f"{count}/{total_number}: {key} - {embedding.shape}")
            paper_embedding_dict[key] = embedding
        count += 1
    return paper_embedding_dict

def process_reviewer_embedding(reviewer_key_value, paper_embedding_dict):
    """为每个审稿人生成嵌入的单线程任务"""
    key, value = reviewer_key_value
    reviewer_record = []
    for each_paper in value:
        if each_paper in paper_embedding_dict.keys():
            reviewer_record.append(paper_embedding_dict[each_paper])

    if reviewer_record:
        reviewer_embedding = np.mean(reviewer_record, axis=0).tolist()
        return key, reviewer_embedding
    return key, None

def get_reviewer_embedding(paper_embedding_dict, reviewer_record_dict):
    """生成审稿人嵌入表示，使用单线程顺序处理"""
    reviewer_embedding_dict = {}
    total_number = len(reviewer_record_dict)

    # 顺序处理每个审稿人的嵌入
    for key, value in tqdm(reviewer_record_dict.items(), total=total_number):
        key, embedding = process_reviewer_embedding((key, value), paper_embedding_dict)
        if embedding is not None:
            reviewer_embedding_dict[key] = embedding

    return reviewer_embedding_dict

def parse_paper_mapping_relationship(filepath):
    paper_mapping_dict = {}
    paper_mapping_list: list = read_list_from_txt(filepath)
    for each_paper_mapping in paper_mapping_list:
        paper_mapping_dict[each_paper_mapping.split(' ')[0]] = each_paper_mapping.split(' ')[1]
    return paper_mapping_dict

def get_reviewer_history(filepath):
    reviewer_record_dict = {}
    review_record_list: list = read_list_from_txt(filepath)
    for each_review_record in review_record_list:
        each_review_record_split = each_review_record.split(' ')
        if len(each_review_record_split) >= 2:
            reviewer_list = each_review_record_split[1:]
            paper_id = each_review_record_split[0]
            for each_reviewer in reviewer_list:
                if each_reviewer not in reviewer_record_dict.keys():
                    reviewer_record_dict[each_reviewer] = [paper_id]
                else:
                    reviewer_record_dict[each_reviewer].append(paper_id)
    return reviewer_record_dict

if __name__ == '__main__':
    paper_filepath = r'get_paper_embedding/dataset_4k/new_paper_info.json'
    reviewer_filepath = r'get_paper_embedding/dataset_4k/new_reviewer_info.json'

    paper_mapping_filepath = r'get_paper_embedding/dataset_4k/submission_list.txt'
    reviewer_history_filepath = r'get_paper_embedding/dataset_4k/train.txt'

    paper_mapping_dict = parse_paper_mapping_relationship(paper_mapping_filepath)
    id_text_dict = load_paper_id_text_from_file(paper_filepath, reviewer_filepath,paper_mapping_dict)
    print('Data loaded')

    # paper_embedding_dict = get_paper_embedding(id_text_dict)
    # print('Paper embedding extracted')
    with open('paper_embeddings.json', 'r') as f:
        paper_embedding_dict = json.load(f)



    # 获取审稿人嵌入
    reviewer_record_dict = get_reviewer_history(reviewer_history_filepath)
    train_reviewer_embedding_dict = get_reviewer_embedding(paper_embedding_dict, reviewer_record_dict)
    print('Reviewer embedding extracted')

    with open('reviewer_embeddings.json', 'w') as f:
        json.dump(train_reviewer_embedding_dict, f, indent=4)
    print('Reviewer embeddings saved to reviewer_embeddings.json')
