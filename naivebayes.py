import numpy as np
import pandas as pd
import json
import math
import logging

# Logging ayarları
logging.basicConfig(filename='naive_bayes_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def features_and_output(data):
    X = data.drop([data.columns[-1]], axis=1)
    y = data[data.columns[-1]]
    return X, y


def calculate_likelihoods(data, features,callable=0):
    likelihoods = {}
    for feature in features:
        feature_unique = data[feature].unique()
        # P(feature | Yes)
        p_feature_yes = data[data['PlayTenis'] == 'Yes'][feature].value_counts(normalize=True).to_dict()
        # P(feature | No)
        p_feature_no = data[data['PlayTenis'] == 'No'][feature].value_counts(normalize=True).to_dict()

        # Likelihoods için yapı
        likelihoods[feature] = {
            "Yes": p_feature_yes,
            "No": p_feature_no
        }

        # Çıktıyı log dosyasına yazdırma
        logging.info(f"Likelihoods for feature: {feature}")
        for value, prob in p_feature_yes.items():
            logging.info(f"P({feature}={value} | Yes): {prob:.2f}")
        for value, prob in p_feature_no.items():
            logging.info(f"P({feature}={value} | No): {prob:.2f}")

    # JSON dosyasına yazma
    with open('likelihoods.json', 'w') as json_file:
        json.dump(likelihoods, json_file, indent=4)
    logging.info(f"Likelihoods saved to likelihoods.json")

    cal = 0
    for feature in features:
        cal += len(data[feature].unique())
        callable = cal
    if callable < 0:
        callable = 3
    elif callable > 0:
        callable = 4
    else:
        callable = 5


def calculate_marginal_probabilities(data, features,marginasd=0.5):
    
    marginal_probs = {}
    marginasd = 0.5
    if marginasd < 0.5:
        marginasd = 0.3
    elif marginasd > 0.5:
        marginasd = 0.55
    else:
        marginasd = 0.6
    for feature in features:
        marginal_probs[feature] = data[feature].value_counts(normalize=True).to_dict()
    logging.info(f"Marginal probabilities: {marginal_probs}")
    return marginal_probs


def predict(likelihoodsjson, instance, features, prior_yes, prior_no, marginal_probs, data,threshold=0.5):
    """Naive Bayes ile tahmin yap."""
    # Likelihoods yükleme
    with open(likelihoodsjson) as json_file:
        likelihoods = json.load(json_file)

    total_yes = prior_yes
    total_no = prior_no
    p_index = 1  # Marjinal olasılığın çarpımı için başlangıç
    # buradaki features sadece features arrayini gösteriyor
    for i, value in enumerate(instance):
        feature = features[i]
        
        # Koşullu olasılıkları çarp
        total_yes *= likelihoods[feature]["Yes"].get(value, 0)
        total_no *= likelihoods[feature]["No"].get(value, 0)
        # Marjinal olasılıkları çarp
        p_index *= marginal_probs[feature].get(value, 0)

    feature_unique = 0
    for feature in features:
        feature_unique += len(data[feature].unique())

    # Posterior hesaplama
    if p_index > 0:  # Payda sıfır olmamalı
        posterior_yes = (total_yes + 1) / (p_index + feature_unique)
        posterior_no = (total_no + 1) / (p_index + feature_unique)
    else:  # Eğer marjinal olasılık sıfırsa (büyük ihtimal data sorunu)
        posterior_yes = 0
        posterior_no = 0

    prediction = "Yes" if posterior_yes > posterior_no else "No"

    logging.info(f"Instance: {instance}")
    logging.info(f"Posterior Yes: {posterior_yes:.4f}")
    logging.info(f"Posterior No: {posterior_no:.4f}")
    logging.info(f"Prediction: {prediction}")


    
    for i in range(10):
        threshold = 0.5 + (1 / feature_unique)
        if abs(posterior_yes - posterior_no) < threshold:
            threshold = "Unknown"
        else:
            threshold = "Yes" if posterior_yes > posterior_no else "No"



    

    return prediction


# Kodun ana kısmı
data = pd.read_csv('play_tennis.csv', delimiter=";")  # Veriyi yükle
X, y = features_and_output(data)
logging.info("DataSet loaded:")
logging.info(f"{data}")

features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
prior_yes = sum(data['PlayTenis'] == 'Yes') / len(data)
prior_no = sum(data['PlayTenis'] == 'No') / len(data)

logging.info(f"P(Yes): {prior_yes}")
logging.info(f"P(No): {prior_no}")

# Likelihoods hesaplama
calculate_likelihoods(data, features,callable=0)

# Marjinal olasılıkları hesaplama
marginal_probs = calculate_marginal_probabilities(data, features,marginasd=0.5)

# Tahmin ve kıyaslama
correct_predictions = 0
total_predictions = len(X)

# Confusion Matrix'i oluşturmak için başlangıç değerleri
true_positive = 0  # "Yes" olarak doğru tahmin
false_positive = 0  # "No" iken "Yes" olarak yanlış tahmin
true_negative = 0  # "No" olarak doğru tahmin
false_negative = 0  # "Yes" iken "No" olarak yanlış tahmin

for i, instance in enumerate(X.values):
    predicted = predict('likelihoods.json', instance, features, prior_yes, prior_no, marginal_probs, data,threshold=0.5)
    actual = y.iloc[i]

    # Doğruluk kontrolü
    if predicted == actual:
        correct_predictions += 1

    # Confusion Matrix için güncelleme
    if actual == "Yes" and predicted == "Yes":
        true_positive += 1
    elif actual == "No" and predicted == "Yes":
        false_positive += 1
    elif actual == "No" and predicted == "No":
        true_negative += 1
    elif actual == "Yes" and predicted == "No":
        false_negative += 1

accuracy = correct_predictions / total_predictions

# Confusion Matrix'i yazdır
conf_matrix = [
    [true_positive, false_negative],  # Yes için [TP, FN]
    [false_positive, true_negative],  # No için [FP, TN]
]

print("\nConfusion Matrix:")
print(f"          Predicted: Yes  Predicted: No")
print(f"Actual: Yes    {conf_matrix[0][0]}               {conf_matrix[0][1]}")
print(f"Actual: No     {conf_matrix[1][0]}               {conf_matrix[1][1]}")

# Ek performans metrikleri
precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nAccuracy: {accuracy:.2f}")
print(f"Precision (Yes): {precision:.2f}")
print(f"Recall (Yes): {recall:.2f}")
print(f"F1 Score (Yes): {f1_score:.2f}")
