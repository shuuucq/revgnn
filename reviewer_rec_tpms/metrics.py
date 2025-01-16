import math

def calculate_ndcg_at_k(recommended, relevant, k):
    dcg = 0.0
    idcg = 0.0
    
    for i in range(min(k, len(recommended))):
        if recommended[i] in relevant:
            dcg += 1.0 / math.log(i + 2, 2)
            
    for i in range(min(k, len(relevant))):
        idcg += 1.0 / math.log(i + 2, 2)
        
    if idcg == 0:
        return 0.0
    else:
        return dcg / idcg

def calculate_recall_at_k(recommended, relevant, k):
    intersection = len(set(recommended[:k]) & set(relevant))
    return intersection / float(len(relevant))

def calculate_hr_at_k(predicted, actual, k=20):
    """
    计算 HR@k。
    
    :param predicted: 推荐的论文列表（按预测分数排序）
    :param actual: 用户实际感兴趣的论文列表
    :param k: 计算的前 k 个推荐项
    :return: HR@k 值
    """
    # 只取前 k 个推荐结果
    predicted = predicted[:k]
    
    # 如果推荐中至少有一个是相关的，HR 为 1，否则为 0
    hits = any(paper_id in actual for paper_id in predicted)
    
    return 1 if hits else 0
def calculate_precision_at_k(predicted, actual, k=20):
    """
    计算 Precision@k。
    
    :param predicted: 推荐的论文列表（按预测分数排序）
    :param actual: 用户实际感兴趣的论文列表
    :param k: 计算的前 k 个推荐项
    :return: Precision@k 值
    """
    # 只取前 k 个推荐结果
    predicted = predicted[:k]
    
    # 计算推荐中有多少是相关的
    relevant_items = set(actual)
    recommended_items = set(predicted)
    
    # 计算推荐中的相关项数量
    relevant_recommendations = len(relevant_items.intersection(recommended_items))
    
    return relevant_recommendations / k
