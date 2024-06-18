**# Recout Machine Learning**
<img src="https://github.com/C241-PS139/Recout-ML/blob/main/image/Logo%20Recout.jpg?raw=true" width="250" height="250"><br /><br />
## What is Recout?
Recout (Recommendation Outfit) is an application designed to meet the outfit needs of the Indonesian community. Recout works by detecting the temperature at the user's location and then providing outfit recommendations that are appropriate for that temperature. Recout emphasizes comfort in fashion, making temperature a key factor in determining what outfit to wear.


## Our Model
This model is a similarity-based recommendation system that uses Cosine Similarity to determine the products that are most similar to user preferences based on porduct gender and temperature in the user's province. The output of this model provides the top 5 most similar products based on product gender and temperature in the user's province, this model uses mathematical and statistical techniques commonly used in machine learning applications. This recommendation system is useful in memory-based collaborative filtering applications, where recommendations are given based on the similarity of user preferences with the dataset we use.

## Dataset
Our dataset can be viewed through [dataset](https://www.kaggle.com/datasets/latifahhukma/fashion-campus/) and [images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset/data).

## Recomendation Model
Our recommendation model was trained with over 14,000+ different fashion data entries focusing on apparels, reaching an impressive precision of 1.00

### Requirements
- pandas
- numpy
- urllib.request
- requests
- cv2
- sklearn
- matplotlib
- joblib

### Recommendation Steps
1. Clone the git repository.
```bash
git clone https://github.com/C241-PS139/Recout-ML.git
```
2. Install the required libraries.
```
pip install pandas numpy urllib3 requests opencv-python scikit-learn matplotlib joblib
```
3. Navigate to the recommendation model directory and open the notebook.
4. Run the cells in the notebook to train the model and evaluate its performance.
5. Save the model into a pickle format on the provided cell.
6. The model can now be used for inferences.
