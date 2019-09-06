#!/bin/bash 
sudo touch miaow.txt
sudo apt-get update 
sudo apt-get install -y htop git python-pip
git clone https://github.com/thejasvibr/the_cocktail_party_nightmare.git
cd the_cocktail_party_nightmare/
git checkout dev_submission2 
#sudo bash cloud_computing_resources/run_script.sh
pip install numpy==1.16.3 scipy==1.2.1 pandas==0.24.2
sudo touch miaow2.txt
cd the_cocktail_party_nightmare
sudo touch bridson/__init__.py
cd simulations/effect_of_group_size_noshadowing/
sudo python group_size_simulations_no_shadowing.py

