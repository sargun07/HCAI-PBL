# import nltk, os
# target = r'C:\Users\VICTUS\Desktop\HCAI\HCAI-PBL\project2\nltk_data'  # absolute path
# os.makedirs(target, exist_ok=True)
# nltk.download('punkt', download_dir=target)
# nltk.download('stopwords', download_dir=target)
# nltk.download('wordnet', download_dir=target)
# nltk.download('averaged_perceptron_tagger', download_dir=target)

# import zipfile, os

# zip_path = r"C:\Users\VICTUS\Desktop\HCAI\HCAI-PBL\project2\nltk_data\corpora\wordnet.zip"
# extract_to = os.path.dirname(zip_path)

# with zipfile.ZipFile(zip_path, 'r') as zf:
#     zf.extractall(extract_to)

# print("Extracted to:", extract_to)

import os
from django.conf import settings
print("BASE_DIR =", settings.BASE_DIR)

data_root = os.path.join(settings.BASE_DIR, "project4", "ml-100k")
print("u.data path:", os.path.join(data_root, "u.data"))
print("exists?", os.path.exists(os.path.join(data_root, "u.data")))
print("u.item path:", os.path.join(data_root, "u.item"))
print("exists?", os.path.exists(os.path.join(data_root, "u.item")))
