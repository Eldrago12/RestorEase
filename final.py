import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import matplotlib.pyplot as plt

file_path = '/content/Sleep_health_and_lifestyle.csv'
dataset = pd.read_csv(file_path)

data = dataset.drop(columns=["Person ID", "Blood Pressure", "Occupation"])

data["BMI Category"] = data["BMI Category"].replace("Normal Weight", "Normal")
bmi_mapping = {"Normal": 0, "Overweight": 1, "Obese": 2}
data["BMI Category"] = data["BMI Category"].map(bmi_mapping)

data["Sleep Disorder"] = data["Sleep Disorder"].replace("None", "Healthy")
sleep_disorder_encoder = LabelEncoder()
data["Sleep Disorder"] = sleep_disorder_encoder.fit_transform(data["Sleep Disorder"])
joblib.dump(sleep_disorder_encoder, "sleep_disorder_encoder.pkl")
data = data[data["Sleep Disorder"].notna()]
y = data["Sleep Disorder"].astype(int)

gender_encoder = LabelEncoder()
data["Gender"] = gender_encoder.fit_transform(data["Gender"])

data.fillna(data.mean(numeric_only=True), inplace=True)

X = data.drop(columns=["Sleep Disorder"])
y = data["Sleep Disorder"]

rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X, y)
feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

# top 5
top_features = feature_importances["Feature"].tolist()[:5]
print(top_features)
X = data[top_features]

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
joblib.dump(scaler, "scaler.pkl")
joblib.dump(top_features, "top_features.pkl")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

X_augmented = X_train + np.random.normal(0, 0.01, X_train.shape)
X_train = np.vstack([X_train, X_augmented])
y_train = np.hstack([y_train, y_train])

print("Train class distribution:", np.bincount(y_train))
print("Test class distribution:", np.bincount(y_test))

# y_train = y_train.astype(int)
# y_test = y_test.astype(int)

print("Unique values in y_train:", np.unique(y_train))
print("Unique values in y_test:", np.unique(y_test))

# y_train = y_train - 1
# y_test = y_test - 1

print("Unique values in y_train:", np.unique(y_train))
print("Unique values in y_test:", np.unique(y_test))

num_classes = len(np.unique(y_train))
print("Number of unique classes:", num_classes)

model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.002)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.002)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu', kernel_regularizer=l2(0.002)),
    Dense(len(np.unique(y_train)), activation='softmax')
])



class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print("Class Weights:", class_weights_dict)

model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    epochs=250,
    batch_size=16,
    validation_split=0.2,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, lr_scheduler]
)

X_test = np.array(X_test)
y_test = np.array(y_test)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

model.save("neural.h5")

top_feature_names = ["BMI Category", "Age", "Sleep Duration", "Physical Activity Level", "Daily Steps"]

feature_indices = [X.columns.get_loc(name) for name in top_feature_names]

X_top_features_train = X_train[:, feature_indices]
X_top_features_test = X_test[:, feature_indices]
joblib.dump(top_feature_names, "top_feature_names.pkl")

rf.fit(X_train, y_train)
joblib.dump(rf, "rf_model.pkl")
rf_probs_train = rf.predict_proba(X_train)
rf_probs_test = rf.predict_proba(X_test)

nn_predictions_train = model.predict(X_train)
nn_predictions_test = model.predict(X_test)

meta_features_train = np.hstack([nn_predictions_train, rf_probs_train, X_top_features_train])
meta_features_test = np.hstack([nn_predictions_test, rf_probs_test, X_top_features_test])

meta_model = LGBMClassifier(
    random_state=42,
    n_estimators=100,
    learning_rate=0.001,
    reg_alpha=2.0,
    reg_lambda=2.0
)
meta_model.fit(meta_features_train, y_train)


sleep_disorder_encoder = joblib.load("sleep_disorder_encoder.pkl")
meta_predictions = meta_model.predict(meta_features_test)

inverse_labels_meta = sleep_disorder_encoder.inverse_transform(meta_predictions)
inverse_labels_nn = sleep_disorder_encoder.inverse_transform(np.argmax(nn_predictions_test, axis=1))
actual_labels = sleep_disorder_encoder.inverse_transform(y_test)

results = pd.DataFrame({
    "Actual": actual_labels,
    "NN Predicted": inverse_labels_nn,
    "Meta Predicted": inverse_labels_meta
})
print(results.head())

np.save("meta_features_train.npy", meta_features_train)
np.save("meta_features_test.npy", meta_features_test)


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Over Epochs')
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')
plt.show()

joblib.dump(meta_model, "optim.pkl")

# Analyze confidence for "Sleep Apnea"
for i, pred in enumerate(nn_predictions_test):
    print(f"Sample {i} NN Probabilities: {pred}, Predicted Class: {np.argmax(pred)}")

print("Raw NN Predictions:", nn_predictions_test[:5])
