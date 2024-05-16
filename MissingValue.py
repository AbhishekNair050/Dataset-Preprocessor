def null_values(data):
    null_columns = data.columns[data.isnull().any()]
    choice = int(input("Enter choice: 1 for drop and 2 to fill "))
    for column in null_columns:
        if choice == 1:
            data = data.drop(column, axis=1)
        elif choice == 2:
            if data[column].dtype == 'object':
                data[column] = data[column].fillna(data[column].mode()[0])
            else:
                data[column] = data[column].fillna(data[column].mean())
    return data

