import subprocess
import os

def print_tree_with_exclusions(start_path, output_file):
    # Exclude __pycache__ directories, all dotfiles/dotdirs, Pipfile, print_project.py, and printed_project.md
    exclude_patterns = ['__pycache__', '.*', 'Pipfile', 'print_project.py', 'printed_project.md']  # Added 'printed_project.md' here
    exclude_args = sum([['-I', pattern] for pattern in exclude_patterns], [])
    
    with open(output_file, 'a') as f:
        subprocess.run(['tree', start_path, '-a'] + exclude_args, stdout=f)

def should_exclude(name):
    # Exclude __pycache__ directories, dotfiles/dotdirs, Pipfile, print_project.py, printed_project.md, and .wav files
    return (name == '__pycache__' or name.startswith('.') or 
            name == 'Pipfile' or name == 'print_project.py' or 
            name == 'printed_project.md' or name.endswith('.wav'))

def print_file_contents(start_path, output_file):
    with open(output_file, 'a') as f:
        for root, dirs, files in os.walk(start_path, topdown=True):
            # Exclude directories and files as specified
            dirs[:] = [d for d in dirs if not should_exclude(d)]
            
            for file in files:
                if should_exclude(file):
                    continue  # Skip excluded files
                
                file_path = os.path.join(root, file)
                if file_path.endswith('print_project.py') or file_path.endswith('printed_project.md'):  # Additional check to exclude these files
                    continue
                f.write(f"\n## {file_path}\n```python\n")  # Adjust the language accordingly
                try:
                    with open(file_path, 'r', encoding='utf-8') as file_content:
                        f.write(file_content.read())
                except Exception as e:
                    f.write(f"Error reading file {file_path}: {e}")
                f.write("\n```\n")

# Clear the output file first in case it already exists and set the start path
output_file = 'printed_project.md'
open(output_file, 'w').close()
start_path = '.'

# Execute the functions to generate the output
print_tree_with_exclusions(start_path, output_file)
print_file_contents(start_path, output_file)
