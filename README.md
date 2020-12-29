# BachelorProef

This project is meant to test if OpenCV could be a lowcost and worthy alternative in possible identification application pipelines. 

To make this project work: 
1. Add folder "dataset" in root of project, 
make a subfolder named the name of the person which you want to train the ML-model with
2. Supply each of these subfolders with pictures of those people (and their faces)
3. Add folder "images" with images of the people you want to recognise. must be .jpg-format
4. First run extract_embeddings.py
5. Run train_model.py to train the model with the extracted faces in the previous step
6. Run recognize.py and see the result. 

A bounding box will be displayed around the picture you want the algorithm to recognise. 
Together with the bounding box it will show the label (name) and the probability of that being that person. 
If the probability is too low a suggestion is to make sure you have enough data.
