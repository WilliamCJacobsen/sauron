# Sauron
> As sauron this project makes your product see everything. So be aware, with great power comes great responsibility.

I use openCV to find and classify faces that I fed it. 

![](https://www.syfy.com/sites/syfy/files/styles/1200x680/public/wire/legacy/sauroneye.png)

## Installation

```sh
pip install pipenv
pipenv install
cd src
```

or
```sh
cd src 
pip install -r requirements.txt
```


## Training
For this you need images of the individuals and place them inside a folder called pictures inside sauron directory. Inside this folder you'll need subfolders with names of the individuals placed inside. You can only use .png or .jpeg/.jpg files. 

To run the training add 
```
if __name__ == "__main__":
    conv = ConvolutionNN(cv2, FACE_CASCADES, 64)
    conv.train();
```
inside main.py


## Usage example

To classify members of a team. 

## Example
```
if __name__ == "__main__":
    conv = ConvolutionNN(cv2, FACE_CASCADES, 64)
    conv.model_summary()
    
    #conv.train();
    the_eye = Sauron(conv)
    
    while True:
        the_eye.recoginze()
```

## Info

Uses CNN to classify the user. The result is softmax values. 
