# Import libraries
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
import ast

warnings.simplefilter('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Function to extract genres from the string representation of the list of dictionaries
def extract_genres(genres_string):
    try:
        genres_list = ast.literal_eval(genres_string)
        return [genre['name'] for genre in genres_list]
    except ValueError:
        return []


# Function to prepare the movie data
def prepare_movie_data(csv_path):
    # Read the dataset
    data = pd.read_csv(csv_path, nrows=5000)

    # Extract and clean the genres
    data['genres'] = data['genres'].apply(extract_genres)

    # Select the relevant columns
    data = data[['original_title', 'overview', 'genres']]

    # Drop rows with missing values in 'overview' or 'genres'
    data.dropna(subset=['overview', 'genres'], inplace=True)

    return data


# Use the function to prepare the data
prepared_data = prepare_movie_data('./archive/movies_metadata.csv')

# Instantiate the binarizer
mlb = MultiLabelBinarizer()

# Fit the binarizer to the genres data - this will find all unique genre labels
mlb.fit(prepared_data['genres'])

# Transform the 'genres' column to a one-hot encoded matrix
genres_encoded = mlb.transform(prepared_data['genres'])

# Create a new DataFrame from the one-hot encoded matrix, with column names set to the unique genre labels
genres_encoded_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

# Join the new genres DataFrame with the original data to include the one-hot encoded genres
movies_with_encoded_genres = prepared_data.join(genres_encoded_df)

# Display the DataFrame
print(movies_with_encoded_genres.head())

# Find the number of occurrences of each genre
genre_counts = np.sum(genres_encoded, axis=0)

class_counts = genres_encoded.sum(axis=0)
# Calculate weights for each class
class_weights = (1. / class_counts) * (len(genres_encoded) / 2.0)
# Convert class weights to a tensor
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss(weight=class_weights_tensor)(outputs, targets)

# Now, calculate the top 20 genres correctly using Pandas
genre_counts_series = pd.Series(genre_counts, index=mlb.classes_)
top_20_genres = genre_counts_series.nlargest(20).index.tolist()

# Create a DataFrame with only the top 20 genres
top_genres_df = movies_with_encoded_genres[top_20_genres]

# Add the 'original_title' and 'overview' columns to the top genres DataFrame
filtered_movies = movies_with_encoded_genres[['original_title', 'overview']].join(top_genres_df)

# Now, we will remove movies that do not have any of the top 20 genres
filtered_movies = filtered_movies[filtered_movies[top_20_genres].sum(axis=1) > 0]

# Print the first few rows of the filtered movies DataFrame
print(filtered_movies.head())

filtered_genre_counts = filtered_movies[top_20_genres].sum().sort_values(ascending=True)
# Create a horizontal bar plot
plt.figure(figsize=(20, 25))
filtered_genre_counts.plot(kind='barh', color='steelblue')
plt.xlabel('Count')
plt.title('Genre Frequency')
plt.show()

# Defining some key variables that will be used later on in the training
MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-06
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)

# Creating the dataset and dataloader for the neural network

train_size = 0.8
train_data=filtered_movies.sample(frac=train_size,random_state=199)
test_data=filtered_movies.drop(train_data.index).reset_index(drop=True)
train_data = train_data.reset_index(drop=True)


class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe['overview']
        self.targets = dataframe[top_20_genres].values
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs.get('token_type_ids')

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long) if token_type_ids is not None else None,
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)

training_set = MultiLabelDataset(train_data, tokenizer, MAX_LEN)
testing_set = MultiLabelDataset(test_data, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 20)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = DistilBERTClass()
model.to(device)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

def train(epoch):
    model.train()
    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids, mask)  # Remove token_type_ids from here

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _ % 5000 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        loss.backward()
        optimizer.step()

for epoch in range(EPOCHS):
    train(epoch)

def validation(testing_loader, movie_titles):
    model.eval()
    fin_targets = []
    fin_outputs = []
    fin_movie_titles = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            fin_movie_titles.extend(movie_titles[i * testing_loader.batch_size : (i + 1) * testing_loader.batch_size])
    return fin_outputs, fin_targets, fin_movie_titles

movie_titles = test_data['original_title'].tolist()

outputs, targets, titles = validation(testing_loader, movie_titles)

# Threshold for converting probabilities to binary predictions
THRESHOLD = 0.3

# Function to apply threshold and convert probabilities to binary values
def apply_threshold(probs, threshold):
    return [[1 if i >= threshold else 0 for i in prob] for prob in probs]

# Applying the threshold to the outputs
binary_outputs = apply_threshold(outputs, THRESHOLD)

# Decode the one-hot encoded labels back to genre names
def decode_genres(encoded_labels, mlb):
    return mlb.inverse_transform(np.array(encoded_labels))

# Decode the true and predicted labels
true_genre_labels = decode_genres(targets, mlb)
predicted_genre_labels = decode_genres(binary_outputs, mlb)

# Create a DataFrame for the true labels and predictions
comparison_df = pd.DataFrame({
    'Movie Title': titles,
    'True Genres': ["|".join(genres) for genres in true_genre_labels],
    'Predicted Genres': ["|".join(genres) for genres in predicted_genre_labels]
})

final_outputs = np.array(outputs) >= THRESHOLD

val_hamming_loss = metrics.hamming_loss(targets, final_outputs)
val_hamming_score = hamming_score(np.array(targets), np.array(final_outputs))

print(f"Hamming Score = {val_hamming_score}")
print(f"Hamming Loss = {val_hamming_loss}")