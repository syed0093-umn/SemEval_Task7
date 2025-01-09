import json
import pprint

# Load the JSON file
with open('tasks.json', 'r') as file:
    tasks = json.load(file)

# Analyze the structure
print("Structure of tasks.json:")
pprint.pprint(tasks, depth=2)

# Count the number of tasks
num_tasks = len(tasks)
print(f"\nNumber of tasks: {num_tasks}")

# Analyze each task
for task_name, task_data in tasks.items():
    print(f"\nAnalyzing task: {task_name}")
    print(f"  Number of fact check IDs: {len(task_data['fact_check_ids'])}")
    print(f"  Number of train post IDs: {len(task_data['train_post_ids'])}")
    print(f"  Number of dev post IDs: {len(task_data['dev_post_ids'])}")

# Analyze language distribution (for monolingual tasks)