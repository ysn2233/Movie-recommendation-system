from urllib.request import urlretrieve
from tqdm import tqdm
import zipfile
import hashlib
import os
import shutil

def extract_file(save_path, _, database_name, data_path):


    print('Extracting {}...'.format(database_name))
    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(data_path)

def download_movielens(dataset_name, data_path):
    """
    Download movielens dataset
    :param dataset_name: which movielen dataset used
    :param data_path: the directory path to place dataset
    """

    url = 'http://files.grouplens.org/datasets/movielens/' + dataset_name + '.zip'
    hash_code = '0e33842e24a9c977be4e0107933c0723'
    extract_path = os.path.join(data_path, dataset_name)
    save_path = os.path.join(data_path, dataset_name + '.zip')

    if os.path.exists(extract_path):
        print('Found {} Data'.format(dataset_name))
        return

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(save_path):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(dataset_name)) as pbar:
            urlretrieve(
                url,
                save_path,
                pbar.hook)

    assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code, \
        '{} file is corrupted.  Remove the file and try again.'.format(save_path)

    os.makedirs(extract_path)
    try:
        extract_file(save_path, extract_path, dataset_name, data_path)
    except Exception as err:
        shutil.rmtree(extract_path)  # Remove extraction folder if there is an error
        raise err

    os.remove(save_path)
    print('Done.')

class DLProgress(tqdm):
    """
    Handle Progress Bar while Downloading
    """
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):

        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

download_movielens('ml-100k', 'dataset')