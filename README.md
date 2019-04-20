# NSFW Model

This repo contains code for running Not Suitable for Work (NSFW) classification.

[online demo](http://ai.midday.me/)

## Usage

#### script 

```bash
python nsfw_predict.py /tmp/test/test.jpeg
```

result : 
```bash
{'class': 'sexy', 'probability': {'drawings': 0.008320281, 'hentai': 0.0011919827, 'neutral': 0.13077603, 'porn': 0.13146976, 'sexy': 0.72824186}}
```

This is a set of scripts that allows for an automatic collection of tens of thousands of images for the following (loosely defined) categories to be later used for training an image classifier:

1.)porn - pornography images.
2.)hentai - hentai images, but also includes pornographic drawings.
3.)sexy - sexually explicit images, but not pornography. Think nude photos, playboy, bikini, etc.
4.)neutral - safe for work neutral images of everyday things and people.
5.)drawings - safe for work drawings (including anime).



Here is what each script (located under scripts directory) does:
1_get_urls.sh - iterates through text files under scripts/source_urls downloading URLs of images for each of the 5 categories above. The Ripme application performs all the heavy lifting. The source URLs are mostly links to various subreddits, but could be any website that Ripme supports. Note: I already ran this script for you, and its outputs are located in raw_data directory. No need to rerun unless you edit files under scripts/source_urls.
2_download_from_urls.sh - downloads actual images for urls found in text files in raw_data directory.
3_optional_download_drawings.sh - (optional) script that downloads SFW anime images from the Danbooru2018 database.
4_optional_download_neutral.sh - (optional) script that downloads SFW neutral images from the Caltech256 dataset
5_create_train.sh - creates data/train directory and copy all *.jpg and *.jpeg files into it from raw_data. Also removes corrupted images.
6_create_test.sh - creates data/test directory and moves N=2000 random files for each class from data/train to data/test (change this number inside the script if you need a different train/test split). Alternatively, you can run it multiple times, each time it will move N images for each class from data/train to data/test.


#### Deploy by TensorFlow Serving

your have to install [Tensorflow Serving](https://www.tensorflow.org/serving/) first

start the server
```bash
./start_tensorflow_serving.sh
```

test server
```bash
python serving_client.py /tmp/test/test.jpeg
```






 
