import os, base64, zipfile, shutil
from pathlib import Path

b64 = open('scratch/updates_b64.txt').read().strip()
code = f'''import os, base64, zipfile, shutil, subprocess, sys
from pathlib import Path

target_dir = Path('/content/group19_midterm_submission')
target_dir.mkdir(parents=True, exist_ok=True)

# 1. Chep ma nguon tu repo_test
if Path('/content/repo_test').is_dir():
    for item in Path('/content/repo_test').iterdir():
        if item.name == '.git': continue
        dest = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

# 2. Giai nen cac ban cap nhat moi nhat
b64_data = """{b64}"""
updates_zip = target_dir / 'updates.zip'
updates_zip.write_bytes(base64.b64decode(b64_data))
with zipfile.ZipFile(updates_zip) as z:
    z.extractall(target_dir)

# 3. Chep dataset chuan tu Google Drive da tai
if Path('/content/Final - PP/dataset').is_dir():
    dest_dataset = target_dir / 'dataset'
    shutil.copytree('/content/Final - PP/dataset', dest_dataset, dirs_exist_ok=True)

print('[OK] Setup hoan tat tai:', target_dir)
print('Cac muc trong submission:', sorted([p.name for p in target_dir.iterdir()]))
'''
open('scratch/colab_setup_code.py', 'w', encoding='utf-8').write(code)
print('Setup code written to scratch/colab_setup_code.py, len:', len(code))
