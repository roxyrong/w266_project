import gdown

URL = "https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ"

output = 'spider.zip'
gdown.download(URL, output, quiet=False)

# unzip spider.zip
# rm spider.zip