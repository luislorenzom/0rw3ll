# 0rw3ll
🧠🤖 Deep learning platform to register faces classify them by several categories by:

* **Gender**: Male or Female
* **Age ranges**: at this moment we're trying to classify by 99 range of years
* **Feelings**:
  * Happiness
  * Surprise
  * Indifference
  * Bewilderment

Also this should hashing the faces to avoid register duplicate data and get recurrency.

## Developer notes
* Download [imdb dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar)
* TODO: Extract info about gender and age from [matlab file]
* Copy result in project/ml/train/dataset
* Train model:
```sh
python main.py -t params.json
```
* Run
```sh
python main.py -m <model_id>
```

## Requirements
* \>= Python3.6
* pip3
* Cuda 10.2
* Nvidia drivers (dev ~> 440.64.00)
* WebCam
* Docker and Docker-Compose