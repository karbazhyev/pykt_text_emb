import pandas as pd
import json
import numpy as np
from bs4 import BeautifulSoup
import math


assist_df = pd.read_csv('/Users/kiakarbasi/pykt_emb_chef/pykt-toolkit-pt_emb/embeddings/data/assist2009/full_dataset/skill_builder_data_corrected_collapsed.csv', encoding = "ISO-8859-1", low_memory=False)
pb_df = pd.read_csv('/Users/kiakarbasi/pykt_emb_chef/pykt-toolkit-pt_emb/embeddings/data/problem_bodies/ProblemBodies_23.csv', low_memory=False)


questions = assist_df['problem_id'].unique()
assist_df['skill_id'].unique()

skill_to_problems = {}
for skill_id in assist_df['skill_id'].unique():
  skill_to_problems[skill_id] = assist_df[assist_df['skill_id'] == skill_id]['problem_id'].unique()


with open('/Users/kiakarbasi/pykt_emb_chef/pykt-toolkit-pt_emb/embeddings/data/assist2009/pb_subset/keyid2idx_assist2009_pb_subset.json', 'r') as f:
  keyid2idx_pb_subset = json.load(f)

problem_ids = list(keyid2idx_pb_subset['questions'].keys())
problem_ids_int = [int(x) for x in problem_ids]
subset_df_html_selected = pb_df[pb_df['problem_id'].isin(problem_ids_int)]
subset_df_html = subset_df_html_selected[['problem_id','problem_body']]



from bs4 import BeautifulSoup

def remove_tags(html):
    soup = BeautifulSoup(html, "html.parser")

    # Find and replace <img> tags with their 'alt' attribute
    for img in soup.find_all('img'):
        alt_text = img.get('alt', '')  # default to empty string if 'alt' is None
        if alt_text:
            # Create a new text node
            new_text = soup.new_string(" " + alt_text + " ")
            img.replace_with(new_text)
        else:
            # Remove the image if no alt text
            img.decompose()

    # Remove all script and style elements
    for tag in soup(['script', 'style']):
        tag.decompose()

    # Extract the text, cleaning up any excessive whitespace
    clean_text = ' '.join(soup.stripped_strings)
    return clean_text

# Example usage with a DataFrame column
# subset_df['problem_body'] = subset_df['problem_body'].apply(remove_tags)




#pass pb_df_subset['problem_body'] column into function remove_tags
subset_df_html['problem_body'] = subset_df_html['problem_body'].apply(remove_tags)