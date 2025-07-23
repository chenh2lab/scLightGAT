#!/bin/bash

echo "Downloading dataset from Google Drive..."
FILE_ID="1JMtC6XfTPpjF7TceN-A9UwIBAlNWnze9"
gdown https://drive.google.com/uc?id=$FILE_ID

echo "Unzipping data.zip..."
unzip data.zip
rm data.zip
echo "✅ Done. Please verify that the 'data/' folder exists."