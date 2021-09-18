#!/bin/bash

mongoexport --uri="$1" --collection=signalfingerprints > all.json
jq -c . < all.json| split -l 1 --additional-suffix=.json -
mkdir jsons
mv *.json jsons/
mv jsons/all.json ./
python3 ./jsons-to-csv.py
rm -r jsons all.json
mv ../dataset/signalfingerprints.csv{,.old$((`ls -1 ../dataset | wc -l` - 1))}
mv ./signalfingerprints.csv ../dataset/
cd ..