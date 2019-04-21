# movie-recommendation-system

The code uses the lightfm recommender system library to train a hybrid content-based + collaborative algorithm that uses the WARP loss function on the movielens dataset. The movielens dataset contains movies and ratings from over 1700 users. Once trained, our script prints out recommended movies for whatever users from the dataset that we choose to terminal.

##Dependencies

numpy (http://www.numpy.org/)

scipy (https://www.scipy.org/)

lightfm (https://github.com/lyst/lightfm)

Install missing dependencies using pip

##Usage

Once you have your dependencies installed via pip, run the script in terminal via



python recommender.py

The blog post related to the code https://towardsdatascience.com/how-to-build-a-movie-recommender-system-in-python-using-lightfm-8fa49d7cbe3b
