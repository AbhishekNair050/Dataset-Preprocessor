def label_encoding(data):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = le.fit_transform(data[column])
    return data

