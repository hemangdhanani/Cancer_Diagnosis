import opendatasets as od
import zipfile
import os
download_url = "https://www.kaggle.com/c/msk-redefining-cancer-treatment"

def download_dataset(url = download_url):
    '''
    Helper function to download the dataset from given url

    Parameter
    =========

    url: str
        url of dataset to be downloaded

    '''
    #od.download(url)
    FOLDER = os.getcwd()
    MFOLDER = os.path.join(FOLDER, "msk-redefining-cancer-treatment")
    TV_FILE = os.path.join(MFOLDER, "training_variants.zip")
    # with zipfile.ZipFile(MFOLDER, 'r') as zip_ref:
    #     zip_ref.extractall(FOLDER)
    with zipfile.ZipFile(TV_FILE, 'r') as zip_ref:
        zip_ref.extractall(FOLDER)
    TT_FILE = os.path.join(MFOLDER, "training_text.zip")
    with zipfile.ZipFile(TT_FILE, 'r') as zip_ref:
        zip_ref.extractall(FOLDER)
    var = os.path.join(FOLDER, "training_variants")
    text = os.path.join(FOLDER, "training_text")
    return var, text

