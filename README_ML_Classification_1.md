My notes of this Machine Learning project

Pre-requisites
    1. Have Dataset
    2. Install packages
        a. pip install pandas scikit-learn
    3. Source Code:
        a. Step 1: Import Libraries
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            import pickle
            
            This step sets up your environment by importing:
                • pandas: For data manipulation and analysis.
                • train_test_split from sklearn.model_selection: To split the dataset into training and testing sets.
                • StandardScaler from sklearn.preprocessing: To standardize features by removing the mean and scaling to unit variance.
                • LogisticRegression from sklearn.linear_model: To create and train a logistic regression model.
                • accuracy_score from sklearn.metrics: To evaluate the accuracy of the model.
                • pickle: To save and load the model and scaler.
            Once you've added these import statements to your script, you're ready to move to the next step. 
            
        Step 2: Load and Preprocess Data
        In this step, you will load your dataset and perform initial preprocessing.
        Since we are using a sample dataset in this example, we'll create the dataset directly in the code. Here is how to load and preprocess the data:
        
        # Sample dataset
            data = {
                'CustomerID': range(1, 11),
                'Age': [25, 34, 22, 45, 35, 50, 29, 41, 38, 30],
                'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
                'AnnualIncome': [50000, 60000, 45000, 80000, 75000, 90000, 62000, 82000, 70000, 65000],
                'CreditScore': [700, 750, 710, 690, 720, 740, 705, 735, 710, 725],
                'Purchased': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']
            }
            
            df = pd.DataFrame(data)
            
            # Display the first few rows of the dataset
            print(df.head())
            
        Explanation:
            • Creating the Dataset: We create a dictionary named data with keys corresponding to column names and values as lists of data points.
            • Loading into DataFrame: We load this dictionary into a pandas DataFrame, df.
            • Display: We print the first few rows of the DataFrame using df.head() to verify the data.
        Run this code to create and display your sample dataset. Once you have done this and verified the output, let me know, and we can proceed to the next step.
        
        Step 3: Encode Categorical Variables
        Here's how to encode the "Gender" and "Purchased" columns:
        
        # Encode Gender column
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
        
        # Encode Purchased column
        df['Purchased'] = df['Purchased'].map({'No': 0, 'Yes': 1})
        
        # Display the first few rows of the encoded dataset
        print(df.head())
        
        Explanation:
            • Encoding Gender: We use the .map() function to convert "Male" to 0 and "Female" to 1.
            • Encoding Purchased: Similarly, we convert "No" to 0 and "Yes" to 1.
            • Display: We print the first few rows of the DataFrame to verify the encoding.
        Run this code to encode the categorical variables and check the updated DataFrame. Once you have done this and verified the output, let me know, and we can proceed to the next step.
        
        
        Step 4: Split the Data into Training and Testing Sets
        In this step, we will split the dataset into features (X) and target (y), and then into training and testing sets.
        Here's how to do it:
        
        # Features and target
        X = df[['Age', 'Gender', 'AnnualIncome', 'CreditScore']]
        y = df['Purchased']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Display the training data
        print("X_train:\n", X_train.head())
        print("y_train:\n", y_train.head())
        
        Explanation:
            • Features (X): We select the columns 'Age', 'Gender', 'AnnualIncome', and 'CreditScore' as our features.
            • Target (y): We select the 'Purchased' column as our target variable.
            • Split Data: We use train_test_split to split the data into training and testing sets. test_size=0.2 means 20% of the data will be used for testing, and random_state=42 ensures reproducibility.
            • Display: We print the first few rows of the training features and target to verify the split.
        Run this code to split your dataset and check the resulting training data. Once you have done this and verified the output, let me know, and we can proceed to the next step.
        
        Step 5: Standardize the Feature Data
        Here's how to standardize the feature data:
        
        # Initialize the StandardScaler
        scaler = StandardScaler()
        
        # Fit and transform the training data
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Transform the testing data
        X_test_scaled = scaler.transform(X_test)
        
        # Display the scaled training data
        print("X_train_scaled:\n", X_train_scaled[:5])
        
        Explanation:
            • Initialize StandardScaler: We create an instance of StandardScaler.
            • Fit and Transform Training Data: We fit the scaler on the training data and then transform it. This ensures that the scaling parameters (mean and standard deviation) are based only on the training data.
            • Transform Testing Data: We transform the testing data using the same scaler. This ensures that the test data is scaled using the same parameters as the training data.
            • Display: We print the first few rows of the scaled training data to verify the standardization.
        Run this code to standardize your feature data and check the scaled training data. Once you have done this and verified the output, let me know, and we can proceed to the next step.
        
        Step 6: Train a Machine Learning Model
        Here's how to train a logistic regression model on the training data:
        # Initialize the logistic regression model
        model = LogisticRegression()
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Display that the model has been trained
        print("Model has been trained.")
        
        Explanation:
            • Initialize Model: We create an instance of LogisticRegression.
            • Train Model: We fit the model on the scaled training data (X_train_scaled) and the training labels (y_train).
        Run this code to train your logistic regression model. Once you have done this and verified that the model has been trained, let me know, and we can proceed to the next step.
        
        Step 7: Evaluate the Model
        We will make predictions on the testing data and then calculate the accuracy of the model.
        Here's how to do it:
        
        
        # Make predictions on the test data
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')
        
        Explanation:
            • Make Predictions: We use the trained model to make predictions on the scaled testing data (X_test_scaled).
            • Calculate Accuracy: We calculate the accuracy of the model by comparing the predicted labels (y_pred) with the true labels (y_test). The accuracy_score function from sklearn.metrics computes the ratio of correctly predicted instances.
        Run this code to evaluate your model and check the accuracy. Once you have done this and verified the accuracy, let me know, and we can proceed to the next step.
        Step 8: Save the Model and Scaler as Pickle Files
        Here's how to save the trained model and the scaler:
        # Save the model
        with open('model.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)
        
        # Save the scaler
        with open('scaler.pkl', 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)
        
        print("Model and scaler have been saved.")
        
        Explanation:
            • Save the Model: We open a file named model.pkl in write-binary mode and use pickle.dump() to save the trained model to this file.
            • Save the Scaler: Similarly, we open a file named scaler.pkl in write-binary mode and use pickle.dump() to save the scaler to this file.
            • Confirmation: We print a message indicating that the model and scaler have been saved.
        Run this code to save your model and scaler. Once you have done this and verified that the files have been saved, let me know, and we can proceed to the final step.
        
        Step 9: Load the Model and Make Predictions
        Here's how to load the saved model and scaler, and make predictions on new data:
        
        # Load the model
        with open('model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        
        # Load the scaler
        with open('scaler.pkl', 'rb') as scaler_file:
            loaded_scaler = pickle.load(scaler_file)
        
        # Example new data
        new_data = pd.DataFrame({
            'Age': [28, 35],
            'Gender': [0, 1],  # 0 for Male, 1 for Female
            'AnnualIncome': [58000, 75000],
            'CreditScore': [720, 710]
        })
        
        # Standardize the new data using the loaded scaler
        new_data_scaled = loaded_scaler.transform(new_data)
        
        # Make predictions
        new_predictions = loaded_model.predict(new_data_scaled)
        
        print("Predictions:", new_predictions)
        
        Explanation:
            • Load the Model: We open the file model.pkl in read-binary mode and use pickle.load() to load the saved model.
            • Load the Scaler: Similarly, we open the file scaler.pkl in read-binary mode and use pickle.load() to load the saved scaler.
            • Prepare New Data: We create a new DataFrame with the same structure as the original features.
            • Standardize New Data: We use the loaded scaler to transform the new data, ensuring it is scaled in the same way as the training data.
            • Make Predictions: We use the loaded model to make predictions on the scaled new data and print the predictions.
        Run this code to load your saved model and scaler, and make predictions on the new data. Once you have done this and verified the predictions, you have completed the entire process! Let me know if you have any questions or need further assistance.
