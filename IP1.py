import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# 1. قراءة البيانات
df = pd.read_csv("heart (1).csv")

# 2. معالجة القيم المفقودة (ملء بالمتوسط)
df = df.fillna(df.mean(numeric_only=True))

# 3. تطبيع القيم الرقمية
scaler = MinMaxScaler()
num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
df[num_features] = scaler.fit_transform(df[num_features])

# 4. حفظ البيانات بعد التنظيف
df.to_csv("cleaned_data.csv", index=False)
print("Data cleaned and saved in cleaned_data.csv")



import seaborn as sns
import matplotlib.pyplot as plt

# قراءة البيانات النظيفة
df = pd.read_csv("cleaned_data.csv")

# 1. ملخص إحصائي
print(df.describe())

# 2. Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap - Correlation Between Features")
plt.show()

# 3. Histogram
df.hist(figsize=(12,10))
plt.suptitle("Histogram of Features")
plt.show()

# 4. Boxplot لكل عمود رقمي
for col in num_features:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for {col}")
    plt.show()


from experta import *

# Define the input facts structure
class HeartRisk(Fact):
    pass

# Define the expert system
class HeartDiseaseExpert(KnowledgeEngine):

    @Rule(HeartRisk(chol=P(lambda x: x > 240), age=P(lambda x: x > 50)))
    def rule_high_chol_age(self):
        print("High Risk: Low maximum heart rate.")

    @Rule(HeartRisk(trestbps=P(lambda x: x > 140), exang=1))
    def rule_high_bp_angina(self):
        print(" High Risk: High blood pressure and exercise-induced angina.")

    @Rule(HeartRisk(thalach=P(lambda x: x < 0.4)))  # normalized value
    def rule_low_thalach(self):
        print(" High Risk: Low maximum heart rate.")

    @Rule(HeartRisk(oldpeak=P(lambda x: x > 0.5)))
    def rule_high_oldpeak(self):
        print(" High Risk: High ST depression (Oldpeak).")

    @Rule(HeartRisk(ca=P(lambda x: x > 0.5)))
    def rule_high_ca(self):
        print(" High Risk: High number of colored vessels (ca).")

    @Rule(HeartRisk(thal=2))
    def rule_thal_fixed_defect(self):
        print(" High Risk: Thal - Fixed Defect.")

    @Rule(HeartRisk(cp=0, age=P(lambda x: x < 0.3)))
    def rule_low_risk_young_cp(self):
        print(" Low Risk: Young age and typical angina.")

    @Rule(HeartRisk(thalach=P(lambda x: x > 0.8), oldpeak=P(lambda x: x < 0.3)))
    def rule_low_risk_thalach(self):
        print(" Low Risk: High heart rate and low ST depression.")

    @Rule(HeartRisk(chol=P(lambda x: x < 200), trestbps=P(lambda x: x < 120)))
    def rule_low_chol_bp(self):
        print(" Low Risk: Normal cholesterol and blood pressure.")

    @Rule(HeartRisk(age=P(lambda x: x < 0.3), exang=0))
    def rule_low_age_exang(self):
        print(" Low Risk: Young and no exercise-induced angina.")

# Sample normalized input (values between 0 and 1)
sample_patient = {
    'age': 0.6,
    'trestbps': 0.7,
    'chol': 0.75,
    'thalach': 0.35,
    'exang': 1,
    'oldpeak': 0.6,
    'ca': 0.6,
    'thal': 2,
    'cp': 3
}

# Run the expert system
engine = HeartDiseaseExpert()
engine.reset()
engine.declare(HeartRisk(**sample_patient))
engine.run()


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. تقسيم البيانات
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. تدريب النموذج
clf = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
clf.fit(X_train, y_train)

# 3. تقييم الأداء
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 4. حفظ النموذج
joblib.dump(clf, "decision_tree_model.joblib")
print(" saved : decision_tree_model.joblib")

