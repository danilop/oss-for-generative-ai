#!/bin/sh

mkdir -p 'data/10k/'

# Download Uber 10-K reports for years 2019 to 2022
for year in {2021..2021}
do
    echo "Downloading Uber 10-K report for $year..."
    curl -s -o "data/10k/uber_${year}.pdf" "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_${year}.pdf"
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded Uber 10-K report for $year"
    else
        echo "Failed to download Uber 10-K report for $year"
    fi
done

# Download VPC Lattice documentation
docs=(
    "https://docs.aws.amazon.com/vpc-lattice/latest/ug/vpc-lattice.pdf"
    "https://docs.aws.amazon.com/vpc-lattice/latest/APIReference/vpc-lattice-api.pdf"
)

for doc in "${docs[@]}"; do
    # Extract the filename from the URL
    filename=$(basename "$doc")
    
    echo "Downloading $filename..."
        
    # Use curl to download the file
    curl -L -s -o "data/$filename" "$doc"

    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded $filename"
    else
        echo "Failed to download $filename"
    fi
done

echo "Download process completed."
