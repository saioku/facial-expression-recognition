# Facial Expression Recognition (FER) model

After successfully training the model, here are the test results:

## Overall performance: 

![Example Image](Model_Results.png)

## Sample of correctly predicted results

![Example Image](Correct_prediction_original.png)

## Sample of incorrectly predicted results

![Example Image](Incorrect_predictions_original.png)


## Validation loss and training loss curve of the original model 

![Example Image](Figure_1.png)

## Validation loss and training loss of the updated model 

![Example Image](Figure_3.png)

## Correct predictions of the updated model

![Example Image](Figure_5.png)

## incorrect predictions of the updated model

![Example Image](Figure_6.png)

## Output of the updated model 

![Example Image](Figure_7.png)

![Example Image](Figure_8.png)



# How to run this Repo

- Clone the repo
    - Type in "git clone https://github.com/saioku/facial-expression-recognition.git" in your terminal 
- Make sure that all of the dependencies are satisfied 
    - ex: if your python environment doesnt recognize the "PIL" import, type in "pip install pillow" or "conda install pillow" based on what your environment is. 
- After all the dependencies are satisfied, run "python main.py"

Note that I used an IDE (VS code) to develop this model if you want to do the same. 


# Changes to the model 

- Data changes: 
    - Took out the images concering lighting and glasses entirely as they were distracting the model
    - These are also not expressions so they are irrelevant to our model's goals.
- Data augmentation: 
    - As we can see from some of the sample images, the data has been augmented ( by flipping, cropping, etc.)
    - Augmenting the data 
- Overall model approach 
    - If we take a look at the loss curves for the original model, we can see that it was really good in identifying trained data, but it was only getting worse in identifying new data (as it was overfitting)
    - It went from essentially one block to multiple blocks, added a lot more layers and regularization, etc.

# Why the model is not perfect 

- Validation loss
    - Although I acheived a general downward trend in validation loss, it is still not low enough compared to the training data. 
    - Is still sporradic
- Validation Accuracy 
    - There is a general upward trend in validation accuracy, but it is still hitting a plateau 
    - Still is sporradic compared to the training accuracy
- High training loss rate
    - Even though the training model accuracy (final acccuracy) is at a 90%+, this suggests that there might be some miscalculation or error in the model. 
- Low data 
    - We cant really do much about this as the dataset is small dataset (unless we add images from another dataset)
- F-1 Score 
    - This is also a low score so hopefully fixing the training/validation loss and accuracy will fix this as well. 

Note that the error might also be on the fundamental level (like the way we measured the model's performance and the model itself). Since the code isnt too long, you guys can try restructing the whole thing as well. 

# Suggested tweaks to the model 

- Change the structure of the model 
    - less/more layers
    - removing/adding more regularixation
- Change parameters
    - Reduce/increase learning rate 
    - add/remove epochss
- Check if the measurements are wrong
    - console log raw data and test 
    - other debugging techniques to make sure that we are measuring the model correctly 

Note that these are only some suggestions, you guys can try something completely different too. 