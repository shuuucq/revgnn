import torch
import torch.optim as optim
from util.loss_torch import bpr_loss
import world
from model import LightGCN, SimGCL
from dataloader import Loader
from util.conf import ModelConf
import json
import numpy as np
from sklearn.metrics import ndcg_score, precision_score, recall_score
import utils  # 确保 utils 模块已正确导入

# Helper function to load SciBERT embeddings from JSON
def load_sci_embeddings(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# Function to combine LightGCN and SciBERT embeddings
def combine_embeddings(lightgcn_emb, scibert_emb):
    combined_emb = torch.cat([lightgcn_emb, scibert_emb], dim=-1)
    return combined_emb

# Function to map real IDs to embeddings
def map_real_id_to_embedding(real_ids, embedding_dict, entity_type='user'):
    """
    Maps real IDs to their corresponding embeddings.
    Args:
        real_ids (list or np.ndarray): List of real IDs.
        embedding_dict (dict): Dictionary mapping IDs to embeddings.
        entity_type (str): Type of entity ('user' or 'item') for debug purposes.
    Returns:
        torch.Tensor: Tensor of embeddings.
    """
    embeddings = []
    for real_id in real_ids:
        key = str(real_id)
        if key in embedding_dict:
            embeddings.append(torch.tensor(embedding_dict[key], dtype=torch.float))
        else:
            print(f"Warning: Missing embedding for {entity_type} ID {real_id}, skipping.")
    if len(embeddings) == 0:
        return torch.empty(0)
    return torch.stack(embeddings).to(world.device)

# Training function
def train_simgcl(data_loader, lightgcn_model, simgcl_model, train_set, reviewer_emb_dict, submission_emb_dict, optimizer):
    lightgcn_model.train()
    simgcl_model.train()

    # Sample the training data
    S = utils.UniformSample_original(data_loader)
    users = torch.Tensor(S[:, 0]).long()  # User IDs
    posItems = torch.Tensor(S[:, 1]).long()  # Positive samples
    negItems = torch.Tensor(S[:, 2]).long()  # Negative samples

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    
    # Shuffle the data
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bce_batch_size'] + 1

    for epoch in range(world.config['train_epochs']):
        total_loss = 0
        batch_count = 0
        for user_idx, pos_idx, neg_idx in utils.next_batch_pairwise(train_set, batch_size=world.config['bce_batch_size']):
            # Convert indices to tensors
            user_idx = torch.tensor(user_idx, dtype=torch.long).to(world.device)
            pos_idx = torch.tensor(pos_idx, dtype=torch.long).to(world.device)
            neg_idx = torch.tensor(neg_idx, dtype=torch.long).to(world.device)

            # Get LightGCN embeddings
            # lightgcn_model.forward should return user_emb and item_emb for given indices
            rec_user_emb, rec_item_emb = lightgcn_model.forward(user_idx, torch.cat([pos_idx, neg_idx]))

            # Split the item embeddings into positive and negative
            pos_rec_item_emb = rec_item_emb[:len(pos_idx)]
            neg_rec_item_emb = rec_item_emb[len(pos_idx):]

            # Get SciBERT embeddings
            user_emb = map_real_id_to_embedding(user_idx.tolist(), reviewer_emb_dict, entity_type='user')
            pos_item_emb = map_real_id_to_embedding(pos_idx.tolist(), submission_emb_dict, entity_type='item')
            neg_item_emb = map_real_id_to_embedding(neg_idx.tolist(), submission_emb_dict, entity_type='item')

            # Check for missing embeddings
            if user_emb.size(0) == 0 or pos_item_emb.size(0) == 0 or neg_item_emb.size(0) == 0:
                print("Skipping this batch due to missing embeddings.")
                continue

            # Combine LightGCN and SciBERT embeddings
            combined_user_emb = combine_embeddings(rec_user_emb, user_emb)
            combined_pos_emb = combine_embeddings(pos_rec_item_emb, pos_item_emb)
            combined_neg_emb = combine_embeddings(neg_rec_item_emb, neg_item_emb)

            # Get SimGCL embeddings for contrastive loss
            user_emb_sim, item_emb_sim = simgcl_model.forward(perturbed=True)

            # Calculate loss: BPR + Contrastive Learning loss
            rec_loss = bpr_loss(combined_user_emb, combined_pos_emb, combined_neg_emb)
            cl_loss = simgcl_model.cal_cl_loss([user_idx, pos_idx])
            loss = rec_loss + cl_loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        average_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch + 1}: Average Loss = {average_loss:.4f}")

# Testing function
def test_simgcl(data_loader, lightgcn_model, simgcl_model, test_set, reviewer_emb_dict, submission_emb_dict, K=10):
    lightgcn_model.eval()
    simgcl_model.eval()

    all_scores = []
    all_labels = []
    top_k_scores = []  # To store Top-K results

    with torch.no_grad():
        for user, items in data_loader.testDict.items():
            user_idx = torch.tensor([user], dtype=torch.long).to(world.device)
            item_indices = torch.tensor(items, dtype=torch.long).to(world.device)

            # Get LightGCN embeddings
            rec_user_emb, rec_item_emb = lightgcn_model.forward(user_idx, item_indices)

            # Get SciBERT embeddings
            user_emb_sci = map_real_id_to_embedding([user], reviewer_emb_dict, entity_type='user')
            item_embs_sci = map_real_id_to_embedding(items, submission_emb_dict, entity_type='item')

            # Check for missing embeddings
            if user_emb_sci.size(0) == 0 or item_embs_sci.size(0) == 0:
                print(f"Skipping user {user} due to missing embeddings.")
                continue

            # Combine LightGCN and SciBERT embeddings
            combined_user_emb = combine_embeddings(rec_user_emb, user_emb_sci)
            combined_item_embs = combine_embeddings(rec_item_emb, item_embs_sci)

            # Compute scores
            scores = torch.matmul(combined_user_emb, combined_item_embs.T).cpu().numpy()

            # Get Top-K items
            top_k_items = np.argsort(scores)[-K:][::-1]
            top_k_scores.append(top_k_items)

            # Store scores and labels
            all_scores.append(scores)
            all_labels.append(np.ones(len(items)))

    # Calculate NDCG, Precision, Recall for Top-K
    # Note: all_labels is a list of arrays of 1s, which may not be correct.
    # Typically, you need to have ground truth labels indicating relevance.
    ndcg = ndcg_score(all_labels, all_scores, k=K)
    precision = precision_score(all_labels, (np.array(all_scores) > 0.5).astype(int), average='macro')
    recall = recall_score(all_labels, (np.array(all_scores) > 0.5).astype(int), average='macro')

    print(f"Top-{K} NDCG: {ndcg:.4f}, Top-{K} Precision: {precision:.4f}, Top-{K} Recall: {recall:.4f}")
    return ndcg, precision, recall

# Main function
def main():
    # Set device
    world.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load training and testing datasets
    data_loader = Loader(config=world.config, path="get_paper_embedding/dataset_4k")

    train_set = list(zip(data_loader.trainUser, data_loader.trainItem))  

    # Convert testDict to list of (user, item)
    test_set = [(user, item) for user, items in data_loader.testDict.items() for item in items]

    # Initialize LightGCN and SimGCL models with training and test sets
    lightgcn_model = LightGCN(conf=ModelConf('conf/LightGCN.yaml'), train_set=train_set, test_set=test_set).to(world.device)
    simgcl_model = SimGCL(conf=ModelConf('conf/SimGCL.yaml'), train_set=train_set, test_set=test_set).to(world.device)

    # Load SciBERT-generated text embeddings
    paper_embedding_dict = load_sci_embeddings('paper_embeddings.json')
    reviewer_embedding_dict = load_sci_embeddings('train_reviewer_embeddings.json')

    # Print sample embeddings to verify
    print(f"Sample reviewer embeddings: {list(reviewer_embedding_dict.items())[:5]}")
    print(f"Sample submission embeddings: {list(paper_embedding_dict.items())[:5]}")

    # Initialize optimizer
    optimizer = optim.Adam(list(lightgcn_model.parameters()) + list(simgcl_model.parameters()), lr=1e-3)

    # Train the models
    train_simgcl(data_loader, lightgcn_model, simgcl_model, train_set, reviewer_embedding_dict, paper_embedding_dict, optimizer)

    # Test the models with specified K value
    test_simgcl(data_loader, lightgcn_model, simgcl_model, test_set, reviewer_emb_dict, paper_embedding_dict, K=10)

if __name__ == '__main__':
    main()
