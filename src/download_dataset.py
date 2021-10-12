import opendatasets as od

# download_url = "https://www.kaggle.com/c/msk-redefining-cancer-treatment"

def download_dataset(url):
    '''
    Helper function to download the dataset from given url

    Parameter
    =========

    url: str
        url of dataset to be downloaded

    '''
    od.download(url)

    # training_variant_zip = './msk-redefining-cancer-treatment/training_variants.zip'

    # !unzip './msk-redefining-cancer-treatment/training_variants.zip' :need to unzip

    # TODO:
    # 1) unzip file
