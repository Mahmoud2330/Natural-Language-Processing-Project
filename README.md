# Natural-Language-Processing-Project

Developed an NLP model to classify restaurant reviews as positive or negative. Using a balanced dataset of 1,000 reviews, text preprocessing and Bag-of-Words transformation were applied. The Multinomial Naive Bayes classifier achieved 86% accuracy. Future improvements include expanding the dataset and refining preprocessing.

## Project Overview

This project focuses on sentiment analysis, a subfield of Natural Language Processing (NLP), to interpret and classify customer reviews for a restaurant as either positive or negative. The main objective is to help the restaurant enhance its customer support strategy by addressing negative reviews promptly and improving overall customer satisfaction.

## Dataset

The dataset consists of 1,000 restaurant reviews, balanced equally between positive and negative sentiments. Each review is labeled with a "Liked" column, where 1 indicates a positive review and 0 indicates a negative review.

## Preprocessing

Preprocessing steps include:
- Converting text to lowercase
- Removing non-alphabetic characters
- Removing extra spaces
- Lemmatization
- Eliminating stopwords

This ensures that the text is clean and standardized, which improves the model's performance.

## Bag-of-Words Representation

The CountVectorizer from scikit-learn was used to transform the preprocessed reviews into a Bag-of-Words format, which converts text data into numerical feature vectors suitable for machine learning models.

## Model Training

The Multinomial Naive Bayes classifier was employed for text classification. The dataset was split into 90% for training and 10% for testing to ensure the model had enough data to learn from while also being evaluated on unseen data.

## Results

The model achieved an accuracy of 86%. The confusion matrix showed 51 true positives, 35 true negatives, 5 false positives, and 9 false negatives, indicating strong predictive performance with room for further improvement.

## Future Improvements

To enhance the model's accuracy and reliability:
- Increase the size of the dataset
- Experiment with more advanced machine learning models
- Refine preprocessing techniques

## Code Summary

The code consists of the following main parts:

1. **Importing Libraries**: Necessary libraries such as pandas, numpy, nltk, scikit-learn, and others are imported.
2. **Loading Data**: The dataset is loaded using pandas.
3. **Preprocessing Function**: A function for preprocessing text data is defined, including lowering case, removing symbols, stop words, and lemmatization.
4. **Bag-of-Words**: The CountVectorizer transforms the text data into a numerical format.
5. **Model Training and Evaluation**: The data is split into training and testing sets, the model is trained using Multinomial Naive Bayes, and its performance is evaluated.

## Contributions

Contributions are welcome! Please feel free to submit a pull request or open an issue to improve the project.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

- Special thanks to the University of Hertfordshire for providing the dataset and resources.
- Thanks to all contributors and the open-source community.
