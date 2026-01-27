#!/usr/bin/env bash
set -x
set -e

rm -rf submit submit.zip
mkdir -p submit

# submit team.txt
printf "Muhammadbager Alali,malali\n, Harry Nguyen, harrycn, Thong Nguyen, thongn3" > submit/team.txt

# train model
python src/myprogram.py train --work_dir work

# make predictions on example data submit it in pred.txt
python src/myprogram.py test --work_dir work --test_data example/input.txt --test_output submit/pred.txt

# submit docker file
cp Dockerfile submit/Dockerfile

# submit source code
cp -r src submit/src

# submit checkpoints
cp -r work submit/work

cp requirements.txt submit/requirements.txt

# make zip file
zip -r submit.zip submit

