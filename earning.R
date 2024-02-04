library(caret)
library(randomForest)
library(e1071)
library(ggplot2)
# Define the file path
# Define the new file path
new_file_path <- "D:/Education/As a Student/NCI Data Analytics/Semester 1/Data Mining and Machine Learning I/Project/Income/census+income (1)/adult.data"

# Read the dataset with the new file path
data <- read.table(new_file_path, header = FALSE, sep = ",", na.strings = "?")


# Show the first few rows of the dataset
head(data)


shape <- dim(data)
# Display the shape
print(shape)


# Check data types in each column
column_types <- sapply(data_cleaned, class)
print(column_types)

# Naming Columns
col_names <- c("age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income")
# Assigning column names to the dataset
colnames(data) <- col_names
# Check the first few rows with column names
head(data)


# Check for missing values in each column
colSums(is.na(data))


# Check for "?" in each column
sapply(data, function(x) sum(x == "?"))

colnames(data)

summary(data)

# Bar plot of the frequency of 'education' levels
ggplot(data, aes(x = education)) +
  geom_bar() +
  labs(title = "Education Level Distribution", x = "Education Level", y = "Frequency")
ggplot(data, aes(x = education, y = hours_per_week)) +
  geom_violin() +
  labs(title = "Hours per Week by Education Level", x = "Education Level", y = "Hours per Week")


# Scatter plot of 'age' against 'capital_gain' colored by 'income'
ggplot(data, aes(x = age, y = capital_gain, color = income)) +
  geom_point() +
  labs(title = "Age vs Capital Gain (Colored by Income)", x = "Age", y = "Capital Gain")

# Density plot of 'fnlwgt'
ggplot(data, aes(x = fnlwgt)) +
  geom_density(fill = "skyblue", color = "black") +
  labs(title = "Density Plot of fnlwgt")




# Select numeric columns
numeric_columns <- sapply(data, is.numeric)
numeric_data <- data[, numeric_columns]

# Create boxplots for numeric variables
par(mfrow = c(3, 2))  # Adjust rows and columns based on your number of numeric variables
for (i in 1:ncol(numeric_data)) {
  boxplot(numeric_data[, i], main = colnames(numeric_data)[i])
}


# Selecting numeric columns
numeric_cols <- sapply(data, is.numeric)
numeric_data <- data[, numeric_cols]

# Calculating z-scores for outliers
z_scores <- scale(numeric_data)
threshold <- 3
outliers <- which(apply(abs(z_scores) > threshold, 1, any))

# Displaying rows with outliers
data[outliers, ]
#Cleaning the rows that contain outliers
data_cleaned <- data[-outliers, ]


head(data_cleaned)

# Identify non-numeric columns
non_numeric_cols <- sapply(data_cleaned, function(col) !is.numeric(col))

# Perform label encoding on non-numeric columns
for (col in names(data_cleaned)[non_numeric_cols]) {
  data_cleaned[[col]] <- as.numeric(factor(data_cleaned[[col]]))
}

# Verify the changes
head(data_cleaned)


head(data_cleaned, 30)


# Selecting numeric columns
numeric_cols <- sapply(data_cleaned, is.numeric)
numeric_data <- data_cleaned[, numeric_cols]

# Compute correlation matrix
correlation_matrix <- cor(numeric_data)

# Print correlation matrix
print(correlation_matrix)

columns_to_drop <- c("education_num", "relationship")
data_selected <- data_cleaned[, !(names(data_cleaned) %in% columns_to_drop)]

# Verify the updated dataset
head(data_selected)

head(data_selected, 30)


numeric_cols <- c("age", "capital_gain", "capital_loss", "hours_per_week")

# Applying Min-Max Scaling
data_selected_scaled <- data_selected
data_selected_scaled[numeric_cols] <- scale(data_selected[numeric_cols])

print(data_selected_scaled)




# Splitting the data into training (70%) and testing (30%) sets
set.seed(123) # For reproducibility
trainIndex <- createDataPartition(data_selected$income, p = 0.7, list = FALSE)
data_train <- data_selected[trainIndex, ]
data_test <- data_selected[-trainIndex, ]

# Convert 'income' to a factor variable with consistent levels in both train and test data
data_train$income <- as.factor(data_train$income)
data_test$income <- factor(data_test$income, levels = levels(data_train$income))

# Train the Naive Bayes model
naive_bayes_model <- naiveBayes(income ~ ., data = data_train)

# Make predictions on the test set
predictions_naive_bayes <- predict(naive_bayes_model, data_test)



# Assess Naive Bayes model performance
confusionMatrix(predictions_naive_bayes, data_test$income)


# Calculate accuracy for Naive Bayes model
accuracy_naive_bayes <- confusionMatrix(predictions_naive_bayes, data_test$income)$overall['Accuracy']

# Update the accuracy data frame
accuracy_df <- data.frame(
  Model = c("Random Forest", "SVM", "Naive Bayes"),
  Accuracy = c(accuracy_rf, accuracy_svm, accuracy_naive_bayes)
)

# Plotting the updated comparison
ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparison of Model Accuracies", y = "Accuracy") +
  theme_minimal()


set.seed(123)
trainIndex <- createDataPartition(data_selected$income, p = 0.7, list = FALSE)
data_train <- data_selected[trainIndex, ]
data_test <- data_selected[-trainIndex, ]

# Convert 'income' to a factor variable with consistent levels in both train and test data
data_train$income <- as.factor(data_train$income)
data_test$income <- factor(data_test$income, levels = levels(data_train$income))

# Train the SVM model
svm_model <- svm(
  income ~ .,
  data = data_train,
  kernel = "radial", # Radial kernel
  scale = TRUE # Feature scaling
)

# Make predictions on the test set
predictions_svm <- predict(svm_model, data_test)

# Assess model performance
confusionMatrix(predictions_svm, data_test$income)

# Calculate accuracy for Random Forest and SVM models
accuracy_rf <- confusionMatrix(predictions, data_test$income)$overall['Accuracy']
accuracy_svm <- confusionMatrix(predictions_svm, data_test$income)$overall['Accuracy']

# Create a data frame for the accuracies
accuracy_df <- data.frame(
  Model = c("Random Forest", "SVM"),
  Accuracy = c(accuracy_rf, accuracy_svm)
)

# Plotting the comparison
ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparison of Model Accuracies", y = "Accuracy") +
  theme_minimal()





