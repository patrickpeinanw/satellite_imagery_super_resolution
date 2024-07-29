### SRCNN Model on Regular Images

This is a preliminary model trained to convert RGB images to higher resolution versions. The purpose of this work is to validate the approaches and methodologies on regular images before applying them to satellite imagery.This is a prelimary model that is trained to convert RBG images to a higher resolution version. The purpose of this work is to validate the approaches and methodologies on a regular images before applying them to satellite imagery.

Both the input and output images have the dimension of 256 \* 256 pixel.

Performance metrics including PSNR, MSE, and SSIM are used to evaluate the model's effectiveness

### Create a Virtual Environment

```sh
python3.8 -m venv venv
```

### Activate the Environment

```sh
source venv/bin/activate
```

### Install Dependencies

```sh
pip install -r requirements.txt
```
