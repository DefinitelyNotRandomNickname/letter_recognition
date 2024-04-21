## About

Simple OCR using tensorflow ML model. It's trained on images of letters in different fonts and styles. You can use it to detect which letter appears on the input image. Before using program make sure you have all the libraries needed from `requirements.txt`.

To run program use `python main.py -i [image]` you can add the flag `--load` for it to load compiled and trained model from the `model` directory. After training the model it's automatically saved in the `model` directory. Additionally you can add the flag `--info` to show more information about distributed percentages.

`create_letters.py` has code to create letters I used as training and testing data. It doesn't contain all the training data tho. 