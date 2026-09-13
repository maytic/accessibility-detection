import json, shutil
from pathlib import Path

root = Path('dataset')

mm_root = root / 'mm'

for split, folder in [('train', 'train2017'), ('val', 'val2017')]:
    split_dir = mm_root / split
    split_dir.mkdir(exist_ok=True, parents=True)

    images_link = split_dir / 'images'

    if not images_link.exists():
        images_link.symlink_to((root / 'images' / folder).resolve())

    src_json = root / f'instances_{folder}_subset.json'
    shutil.copyfile(src_json, split_dir / 'labels.json')


    data = json.loads((split_dir / 'labels.json').read_text())

    print(split, len(data['images']), 'images,', len(data['annotations']), 'annotations')