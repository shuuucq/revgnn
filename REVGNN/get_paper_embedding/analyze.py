import utils
from data.ui_graph import Interaction
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm  # Use tqdm suitable for Jupyter Notebook
import os

# Load data
if not os.path.exists('/home/aimingshu/project/Reviewer-Rec-main/dataset_8k/interaction_8.txt'):
    print("File not found!")

all_set = utils.load_data('get_paper_embedding/dataset_4k/interaction_8.txt')
val_set = utils.load_data('get_paper_embedding/dataset_4k/valid.txt')
test_set = utils.load_data('get_paper_embedding/dataset_4k/test.txt')
interaction = Interaction(None, all_set, val_set, test_set)

# Count the number of external user IDs for each external item ID
user_count_distribution = defaultdict(int)
item_id_distribution = defaultdict(list)  # Store item IDs corresponding to each user count

for item_id in tqdm(interaction.item.values(), desc="Processing items"):
    external_item_id = interaction.id2item[item_id]
    external_user_ids, _ = interaction.item_rated(external_item_id)
    user_count = len(external_user_ids)  # Count the number of users
    user_count_distribution[user_count] += 1  # Increment the count for this user count
    item_id_distribution[user_count].append(external_item_id)  # Record the corresponding item ID

# Prepare the data for plotting
user_counts = list(user_count_distribution.keys())
frequencies = list(user_count_distribution.values())

# Print the maximum user count and the corresponding external item IDs
max_user_count = max(user_counts) if user_counts else 0
print(f"Maximum user count: {max_user_count}")

# Get all external item IDs corresponding to the maximum user count
if max_user_count in item_id_distribution:
    corresponding_items = item_id_distribution[max_user_count]
    print(f"Corresponding external item IDs: {corresponding_items}")

# Plot the statistics
plt.figure(figsize=(12, 6))
bars = plt.bar(user_counts, frequencies, color='skyblue')

# Print the exact counts on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

plt.xlabel('Number of Reviews')  # Change x-axis label
plt.ylabel('Number of People')    # Change y-axis label
plt.title('Count of External Item IDs by Number of External User IDs')
plt.xticks(user_counts)  # Show only the user counts that appeared
plt.tight_layout()  # Automatically adjust layout to prevent overlap

# Save the image
plt.savefig('4k_distribution.png', format='png')  # Save as PNG format
plt.show()
