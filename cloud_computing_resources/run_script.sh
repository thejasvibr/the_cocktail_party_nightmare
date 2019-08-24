#!/bin/bash
pip install numpy==1.16.3 scipy==1.2.1 pandas==0.24.2
cd the_cocktail_party_nightmare
sudo touch bridson/__init__.py
cd simulations/effect_of_group_size_noshadowing/
python group_size_simulations_no_shadowing.py
