import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from data_preprocess.util_data_preprocess import read_json_file, read_list_from_txt

# Load model directly
local_model_path = "/md0/home/aimingshu/project/Reviewer-Rec-main/sci-bert"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModel.from_pretrained(local_model_path)

def load_paper_id_text_from_file(paper_filepath, reviewer_filepath):
    """根据文章id获取相关信息"""
    id_text_dict = {}
    paper_detail = read_json_file(paper_filepath)
    reviewer_detail = read_json_file(reviewer_filepath)

    for each_paper in paper_detail:
        id_text_dict[each_paper['_id']] = {
            'title': each_paper['Title'],
            'abstract': each_paper.get('Abstract', ''),
            "keywords": each_paper.get('Keywords', []),
            "venue": each_paper.get('JournalTitle', ''),
            "authors": each_paper.get('Authors', [])
        }

    return id_text_dict

def get_sci_emb(input_dict):
    """将文本信息进行嵌入表示"""
    title = input_dict['title']
    abstract = input_dict.get('abstract', '')

    keywords = input_dict.get('keywords', [])
    if not isinstance(keywords, list):  # 确保 keywords 是一个列表
        keywords = []

    keywords_str = ', '.join(keywords)  # 将关键词连接成字符串
    venue = input_dict.get('venue', '')

    authors = [author['FullName'] for author in input_dict.get('authors', []) if isinstance(author, dict) and 'FullName' in author]
    authors = ', '.join(authors)

    # 合并所有输入
    combined_text = f"{title} [SEP] {abstract} [SEP] {keywords_str} [SEP] {venue} [SEP] {authors}"

    # 进行分词并创建输入张量，同时进行截断
    inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # 从 SciBERT 获取嵌入表示
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]  # 提取 [CLS] token 的嵌入

    return embedding.flatten().cpu().tolist()  # 将 Tensor 转换为 Python 列表

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
    """解析论文 ID 映射关系"""
    paper_mapping_dict = {}
    paper_mapping_list = read_list_from_txt(filepath)
    for each_paper_mapping in paper_mapping_list:
        paper_mapping_dict[each_paper_mapping.split(' ')[0]] = each_paper_mapping.split(' ')[1]
    return paper_mapping_dict

def get_reviewer_history(filepath):
    """获取审稿人历史信息"""
    reviewer_record_dict = {}
    review_record_list = read_list_from_txt(filepath)
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
    submission_history = r'get_paper_embedding/dataset_4k/submission_list.txt'
    reviewer_history_filepath = r'get_paper_embedding/dataset_4k/train.txt'
    
    # 解析映射和加载论文与审稿人数据
    paper_mapping_dict = parse_paper_mapping_relationship(submission_history)
    id_text_dict = load_paper_id_text_from_file(paper_filepath, reviewer_filepath)
    print('Data loaded')

    # 如果已保存嵌入，直接加载；否则，生成并保存论文嵌入
    with open('paper_embeddings.json', 'r') as f:
        paper_embedding_dict = json.load(f)

    # 获取审稿人嵌入
    # reviewer_record_dict = get_reviewer_history(reviewer_history_filepath)
    # train_reviewer_embedding_dict = get_reviewer_embedding(paper_embedding_dict, reviewer_record_dict)
    # print('Reviewer embedding extracted')

    # with open('train_reviewer_embeddings.json', 'w') as f:
    #     json.dump(train_reviewer_embedding_dict, f, indent=4)
    # print('Reviewer embeddings saved to reviewer_embeddings.json')
