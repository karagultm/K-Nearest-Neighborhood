import numpy as np
import pandas as pd
import json

#bu hatalı kod bu değiştirlicek 

def features_and_output(data):
    X = data.drop([data.columns[-1]], axis=1)
    y = data[data.columns[-1]]
    return X, y

def encode_features(data):
    #One hot encodeing for categorical data
    encoding_maps = {
        'Outlook': {'Sunny': 1, 'Overcast': 2, 'Rain': 3},
        'Temperature': {'Hot': 1, 'Mild': 2, 'Cool': 3},
        'Humidity': {'High': 1, 'Normal': 2},
        'Wind': {'Weak': 1, 'Strong': 2}
    }
    
    # Her bir sütunu kodlama
    for feature, mapping in encoding_maps.items():
        if feature in data.columns:
            data[feature] = data[feature].map(mapping)
    return data

def get_user_input():
    #taking neighbour value from user
    while True:
        try:
            k = int(input("Lütfen k (komşu sayısı) değerini girin: "))
            if k > 0:
                break
            else:
                print("k değeri pozitif bir sayı olmalıdır.")
        except ValueError:
            print("Lütfen geçerli bir tam sayı girin.")

    #taking distance metric from user
    while True:
        try: 
            print("\nMesafe metriğini seçin:")
            print("1. Euclidean Distance")
            print("2. Manhattan Distance")
            choice = input("Seçiminizi yapın (1 veya 2): ")
            match choice:
                case "1":
                    distance_metric = "euclidean"
                    break
                case "2":
                    distance_metric = "manhattan"
                    break
                case _:
                    print("Lütfen geçerli bir seçim yapın (1 veya 2).")
            # if choice == "1":
            #     distance_metric = "euclidean"
            #     break
            # elif choice == "2":
            #     distance_metric = "manhattan"
            #     break
            # else:
            #     print("Lütfen geçerli bir seçim yapın (1 veya 2).")
        except ValueError:
            print("Lütfen geçerli bir seçim yapın (1 veya 2).")

    return k, distance_metric

def calculate_distances(X_train, X_test, metric):
    distances = []

    if metric == "euclidean":
        # Euclidean mesafe hesaplama
        for _, row in X_train.iterrows():
            dist = np.sqrt(np.sum((row - X_test) ** 2))
            distances.append(dist)
    elif metric == "manhattan":
        # Manhattan mesafe hesaplama
        for _, row in X_train.iterrows():
            dist = np.sum(np.abs(row - X_test))
            distances.append(dist)

    # Mesafeleri bir pandas Series olarak döndür
    return pd.Series(distances, index=X_train.index)
 

def loocv_test(data, k, metric):
    correct_predictions = 0
    total_predictions = len(data)
    # Confusion Matrix'i oluşturmak için başlangıç değerleri
    true_positive = 0  # "Yes" olarak doğru tahmin
    false_positive = 0  # "No" iken "Yes" olarak yanlış tahmin
    true_negative = 0  # "No" olarak doğru tahmin
    false_negative = 0  # "Yes" iken "No" olarak yanlış tahmin
    for i in range(len(data)):
        # Test verisini ayır
        test_data = data.iloc[i]
        train_data = data.drop(i)

        # Özellikleri ve çıktıyı ayır
        X_train, y_train = features_and_output(train_data)
        X_test, y_test = features_and_output(pd.DataFrame([test_data]))

        # Mesafeleri hesapla
        distances = calculate_distances(X_train, X_test.iloc[0], metric)
        

        # En yakın k komşuyu bul
        nearest_neighbors = distances.nsmallest(k).index

        nearest_labels = y_train.loc[nearest_neighbors]

        # Sınıfı tahmin et
        prediction = nearest_labels.mode()[0]
        # print(f"\nTest Verisi: {test_data}")
        # print(f"Tahmin: {prediction}")
        # print(f"True Positive: {true_positive}")
        # print(f"False Positive: {false_positive}")
        # print(f"True Negative: {true_negative}")
        # print(f"False Negative: {false_negative}")
        
        # Tahminin doğruluğunu kontrol et
        if prediction == y_test.iloc[0]:
            correct_predictions += 1
        # Confusion Matrix için güncelleme
        if y_test.iloc[0] == "Yes" and prediction == "Yes":
            true_positive += 1
        elif y_test.iloc[0] == "No" and prediction == "Yes":
            false_positive += 1
        elif y_test.iloc[0] == "No" and prediction == "No":
            true_negative += 1
        elif y_test.iloc[0] == "Yes" and prediction == "No":
            false_negative += 1
            
    accuracy = (correct_predictions / total_predictions) 
    
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

def save_model(data,encoded_data, k_value, distance_metric):
    try:
        model= {
            "data": data.to_dict(orient='records'),
            "encoded_data": encoded_data.to_dict(orient='records'),
            "k_value": k_value,
            "distance_metric": distance_metric
        }

        with open('knn_model.json', 'w') as file:
            json.dump(model, file,indent=4)
    except Exception as e:
        print(f"Saving Model Exception Message: {e}")



#Main Code Block starting from here
data = pd.read_csv('play_tennis.csv', delimiter=";")
# print("Orijinal Veri Seti:")
# print(data)

# Özellikleri encode et
encoded_data = encode_features(data)
# print("\nKodlanmış Veri Seti:")
# print(encoded_data)

# Özellik ve çıktı kolonlarını ayır
X, y = features_and_output(encoded_data)
# print("\nÖzellikler:")
# print(X)
# print("\nÇıktı:")
# print(y)

# Kullanıcıdan k ve mesafe metriğini al
k_value, distance_metric = get_user_input()
save_model(data,encoded_data,k_value,distance_metric) #saving model to json file
loocv_test(data, k_value, distance_metric)