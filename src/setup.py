import os

# required directories for project
folders = [
    "data/raw",
    "data/processed",
    "models",
    "notebooks",
    "src"
]

# create directories if they don't exist
for folder in folders:
    os.makedirs(folder, exist_ok=True)
