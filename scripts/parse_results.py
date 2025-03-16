import json

# this script is used to fix the missing braces and quotes in the json file outputs. 
# Some outputs have been formatted incorrectly by the LLM and this file contains differnt resolution strategies
# the functions in this file are called in visualization.ipynb to fix the json files before they are used for visualization

FIX_COMBOS = [
    "",      
    "}",      
    "\"",     
    "\"}",    
    "}\"",    
    "\"}\""   
]

def try_fix_json(json_string):
    """
    Tries parsing 'json_string' with various appended fix combos.
    Returns the first successfully parsed object or None if none worked.
    """
    for combo in FIX_COMBOS:
        candidate = json_string + combo
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return None  # Unable to parse with any appended combos

def fix_line_outer_and_review(line):
    """
    1) Attempt to parse the entire line as JSON (outer).
    2) If that fails, try appending possible combos (} and ") until it works.
    3) Then parse the 'review' key as JSON (inner). If that fails,
       append combos again.
    4) If both parse, re-embed the fixed 'review' as a proper string into
       the outer object, and return a valid JSON string for the entire line.
    5) Return None if we can’t fix it.
    """
    line = line.rstrip("\n")
    if not line.strip():
        return None

    # 1) Fix the outer JSON
    outer_data = try_fix_json(line)
    if outer_data is None:
        return None  # Can't fix outer JSON

    # 2) We have a dict; now fix the 'review' field if it exists
    if "review" not in outer_data:
        # If there's no 'review' field, just re-dump
        return json.dumps(outer_data)

    review_str = outer_data["review"]
    if not isinstance(review_str, str):
        # If 'review' isn't a string, nothing to fix
        return json.dumps(outer_data)

    # 3) Try to parse the review_str as JSON
    fixed_inner_data = try_fix_json(review_str)
    if fixed_inner_data is None:
        # Could not parse even after appending combos
        return None

    # 4) Re-embed the fixed 'review' data as a string
    outer_data["review"] = json.dumps(fixed_inner_data, ensure_ascii=False)

    # 5) Return a valid JSON line
    return json.dumps(outer_data, ensure_ascii=False)


def fix_missing_quotes_and_braces(input_file, output_file):
    """
    Reads each line of 'input_file', attempts to repair truncated JSON 
    by adding up to a couple of braces/quotes at the end, then writes
    out the fixed (or original if no fix needed) lines to 'output_file'.
    Skips lines we cannot fix.
    """
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        all_fixed_lines = []
        for i, line in enumerate(fin, start=1):
            original_line = line.rstrip("\n")
            fixed_line = fix_line_outer_and_review(original_line)
            if fixed_line is None:
                print(f"Line {i}: Could not fix JSON by appending braces/quotes. Skipping.\n"
                      f"  Original line: {original_line}\n")
                continue

            fout.write(fixed_line + "\n")
            all_fixed_lines.append(fixed_line)
    return all_fixed_lines

