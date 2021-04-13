import requests
import gzip
import os, shutil
from zipfile import ZipFile

url = 'https://storage.googleapis.com/codehub-data/7-2P-dataset_renamed.zip'

import urllib.request

from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

    with ZipFile(output_path, 'r') as zipped:
        zipped.printdir()

        print('Extracting all files')
        zipped.extractall()
        print('Done!')


if __name__=='__main__':
    download_url(output_path='dataset.zip',url=url)

