import re

# Read the app.py file
with open('app.py', 'r', encoding='utf-8') as file:
    content = file.read()

# Find all occurrences of the problematic key
key_pattern = r'key="individual_analysis_display_option_1"'
matches = list(re.finditer(key_pattern, content))

print(f"Found {len(matches)} occurrences of the duplicate key")

# Replace only the second occurrence (and onwards) with a different key
if len(matches) > 1:
    # Start from the second occurrence (index 1)
    for i in range(1, len(matches)):
        match = matches[i]
        start, end = match.span()
        # Replace with a numbered key
        new_key = f'key="individual_analysis_display_option_{i+1}"'
        content = content[:start] + new_key + content[end:]
        print(f"Replaced occurrence {i+1} with {new_key}")

# Write the fixed content back to the file
with open('app.py', 'w', encoding='utf-8') as file:
    file.write(content)

print("Fixed the duplicate key issues in app.py")
