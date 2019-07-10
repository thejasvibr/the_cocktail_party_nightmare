#!/bin/bash 
sudo apt-get install -y htop git python-pip
git clone https://github.com/thejasvibr/the_cocktail_party_nightmare.git
cd the_cocktail_party_nightmare/
git checkout dev_submission2 
sudo bash cloud_computing_resources/run_script.sh
