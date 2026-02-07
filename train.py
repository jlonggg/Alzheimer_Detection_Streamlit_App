import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score,accuracy_score,confusion_matrix,recall_score, precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

df=pd.read_csv('alzheimers_prediction_dataset.csv')
df.head(5)

#Tiền xử lí sơ bộ
print(df.isnull().sum())
le=LabelEncoder()
for col in df.columns:
    if df[col].nunique()==2:
        df[col]=le.fit_transform(df[col])
map={'Low':0, 'Medium':1, 'High':2}
df['Physical Activity Level']=df['Physical Activity Level'].map(map)
df['Depression Level']=df['Depression Level'].map(map)
df['Air Pollution Exposure']=df['Air Pollution Exposure'].map(map)
df['Social Engagement Level']=df['Social Engagement Level'].map(map)
df['Income Level']=df['Income Level'].map(map)
df['Stress Levels']=df['Stress Levels'].map(map)
map={'Never':0,'Former':1,'Current':2}
df['Smoking Status']=df['Smoking Status'].map(map)
map={'Poor':0,'Average':1,'Good':2}
df['Sleep Quality']=df['Sleep Quality'].map(map)
map={'Never':0,'Occasionally':1,'Regularly':2}
df['Alcohol Consumption']=df['Alcohol Consumption'].map(map)
map={'Unhealthy':0,'Average':1,'Healthy':2}
df['Dietary Habits']=df['Dietary Habits'].map(map)
map={'Unemployed':0,'Retired':1,'Employed':2}
df['Employment Status']=df['Employment Status'].map(map)
map={'Single':0,'Widowed':1,'Married':2}
df['Marital Status']=df['Marital Status'].map(map)
print(df.head(5))
print(df.isnull().sum())
country_to_continent = {
    # Châu Âu
    'Spain': 'Europe',
    'Sweden': 'Europe',
    'Germany': 'Europe',
    'UK': 'Europe',
    'Italy': 'Europe',
    'France': 'Europe',
    'Norway': 'Europe',
    'Russia': 'Europe',

    # Bắc Mỹ
    'USA': 'North America',
    'Canada': 'North America',
    'Mexico': 'North America' ,

    # Nam Mỹ
    'Argentina': 'South America',
    'Brazil': 'South America',

    # Châu Á
    'China': 'Asia',
    'India': 'Asia',
    'Japan': 'Asia',
    'South Korea': 'Asia',
    'Saudi Arabia': 'Asia',

    # Châu Phi
    'South Africa': 'Africa',

    # Châu Úc/Châu Đại Dương
    'Australia': 'Oceania',


}

df['Country'] = df['Country'].map(country_to_continent)
col='Country'
dummies=pd.get_dummies(df[col],prefix=col)
df=pd.concat([df,dummies],axis=1)
df.drop(col,axis=1,inplace=True)
df.head()

age_bins = [0,66,76,100]
age_labels = ['65-','66-75','76+']

# Áp dụng binning
df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
le=LabelEncoder()
df['age_group']=le.fit_transform(df['age_group'])
print(df.head())

X=df[['Age','Family History of Alzheimer’s','Genetic Risk Factor (APOE-ε4 allele)','BMI','Cognitive Test Score','Education Level',
      'Physical Activity Level',
'Smoking Status',
'Alcohol Consumption',
'Diabetes',
'Hypertension',
'Cholesterol Level',
'Depression Level',
'Sleep Quality',
'Dietary Habits',
'Air Pollution Exposure',
'Employment Status',
'Marital Status',
'Gender',
'Social Engagement Level',
'Income Level',
'Stress Levels',

'Urban vs Rural Living',
      'Country_Africa','Country_Europe','Country_North America','Country_South America']]

X_final=X.drop(['Income Level','Employment Status','Marital Status','Cognitive Test Score','Education Level','Air Pollution Exposure','Dietary Habits','Smoking Status'],axis=1)
y=df['Alzheimer’s Diagnosis']

dict_group = {
    '0-5%':[],
    '5-10%':[],
    '10-15%':[],
    '15-20%':[],
    '20-25%':[],
    '25-30%':[],
    '30-35%':[],
    '35-40%':[],
    '40-45%':[],
    '45-50%':[],
    '50-55%':[],
     '55-60%':[],
      '60-65%':[],
     '65-70%':[],
    '70-75%':[],
    '75-80%':[],
    '80-85%':[],
    '85-90%':[],
    '90-95%':[],
    '95-100%':[]
}

X_train,X_test,y_train,y_test=train_test_split(X_final,y,test_size=0.2,random_state=13)
healthy_count = sum(y_train == 0)
disease_count = sum(y_train == 1)
imbalance_ratio = healthy_count / disease_count
from xgboost import XGBClassifier
model_final=XGBClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
     scale_pos_weight=imbalance_ratio ,
    eval_metric='logloss'
)
model_final.fit(X_train,y_train)
y_pred = model_final.predict(X_test)

from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(
    model_final,
    method='isotonic',
    cv=5
)
calibrated_model.fit(X_train, y_train)

X_test['pred_prob']=calibrated_model.predict_proba(X_test)[:,1]
bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%', '30-35%', '35-40%', '40-45%', '45-50%',
          '50-55%', '55-60%',
           '60-65%', '65-70%','70-75%', '75-80%', '80-85%','85-90%','90-95%','95-100%']

X_test['pred_group'] = pd.cut(X_test['pred_prob'], bins=bins, labels=labels)
X_test['target']=y_test
  #Tính toán từng bin
all_groups = X_test['pred_group'].dropna().unique()
for group in sorted(all_groups):
    # Lọc ra những người trong nhóm này
      mask = X_test['pred_group'] == group
      group_data = X_test[mask]

    # Tính:
      n_people = len(group_data)  # Số người trong nhóm
      avg_pred = group_data['pred_prob'].mean()  # Model dự đoán trung bình bao nhiêu %

    # % THỰC TẾ: bao nhiêu % người trong nhóm có target=1
    # Công thức: (số người target=1) / (tổng số người)
      n_target_1 = group_data['target'].sum()  # Số người target=1 (vì target=0/1)
      actual_pct = n_target_1 / n_people if n_people > 0 else 0

    # Tính sai số
      error = abs(avg_pred - actual_pct)
      dict_group[group].append(error)

dict_mean={}
for col in dict_group.keys():
  dict_mean[col]=np.mean(dict_group[col])*100
print(pd.DataFrame([dict_mean]))

mean_value = np.nanmean(list(dict_mean.values()))
print(mean_value)

# Cell from LLM
# ===== CELL 6: Train final calibrated model on FULL data + save PKL =====
import pickle as pkl
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import sklearn, xgboost

# 1) Chuẩn hoá X_final, y (chống NaN)
X_all = X_final.copy()
X_all = X_all.apply(pd.to_numeric, errors="coerce").fillna(0)

y_all = y.copy()
y_all = pd.to_numeric(y_all, errors="coerce").fillna(0).astype(int)

# đảm bảo y chỉ có 0/1
uniq = sorted(pd.unique(y_all))
print("y unique:", uniq)
if not set(uniq).issubset({0, 1}):
    raise ValueError(f"y không phải 0/1, đang có: {uniq}")

# 2) imbalance ratio tính trên TOÀN BỘ data
healthy_count = int((y_all == 0).sum())
disease_count = int((y_all == 1).sum())
imbalance_ratio = healthy_count / max(disease_count, 1)
print("healthy:", healthy_count, "disease:", disease_count, "ratio:", imbalance_ratio)

# 3) XGB base model
xgb_base = XGBClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=imbalance_ratio,
    eval_metric="logloss",
    random_state=13,
)

# 4) CalibratedClassifierCV (tương thích nhiều sklearn version)
try:
    cal_model = CalibratedClassifierCV(estimator=xgb_base, method="isotonic", cv=5)
except TypeError:
    cal_model = CalibratedClassifierCV(base_estimator=xgb_base, method="isotonic", cv=5)

# 5) Fit trên full data
cal_model.fit(X_all, y_all)

# 6) Tạo maps cho deploy (tự khai báo lại cho chắc chắn, KHÔNG phụ thuộc biến "map" bị overwrite)
maps = {
    "map_level": {'Low':0, 'Medium':1, 'High':2},
    "map_smoking": {'Never':0,'Former':1,'Current':2},
    "map_sleep": {'Poor':0,'Average':1,'Good':2},
    "map_alcohol": {'Never':0,'Occasionally':1,'Regularly':2},
    "map_diet": {'Unhealthy':0,'Average':1,'Healthy':2},
    "map_employment": {'Unemployed':0,'Retired':1,'Employed':2},
    "map_marital": {'Single':0,'Widowed':1,'Married':2},
}

# mapping nhị phân hợp lý theo LabelEncoder alphabet (No<Yes, Female<Male, Rural<Urban)
binary_maps = {
    "Family History of Alzheimer’s": {"No":0, "Yes":1},
    "Diabetes": {"No":0, "Yes":1},
    "Hypertension": {"No":0, "Yes":1},
    "Gender": {"Female":0, "Male":1},
    "Urban vs Rural Living": {"Rural":0, "Urban":1},
    # Genetic Risk Factor bạn nhập 0/1 luôn
}

# 7) Lưu artifact
artifact = {
    "model": cal_model,
    "features": X_all.columns.tolist(),      # đúng thứ tự cột
    "country_to_continent": country_to_continent,
    "maps": maps,
    "binary_maps": binary_maps,
    "versions": {
        "python": sys.version,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
}

with open("alzheimers_artifact.pkl", "wb") as f:
    pkl.dump(artifact, f)

print("✅ Saved -> alzheimers_artifact.pkl")
print("n_features:", len(artifact["features"]))
print("versions:", artifact["versions"])