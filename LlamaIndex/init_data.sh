#!/bin/sh
mkdir -p 'data/10k/'
# Download Uber 10-K reports for years 2019 to 2022
for year in {2019..2022}
do
    echo "Downloading Uber 10-K report for $year..."
    curl -o "data/10k/uber_${year}.pdf" "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_${year}.pdf"
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded Uber 10-K report for $year"
    else
        echo "Failed to download Uber 10-K report for $year"
    fi
done
