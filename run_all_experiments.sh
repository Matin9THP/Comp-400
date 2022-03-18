# training of base model
python src/main.py -model_dir ./models/base_model -mode train_eval

# moving the checkoints to the base_model folder
mv ./checkpoints ./models/base_model/

# training of base model with no batch normalization
python src/main.py -model_dir ./models/base_model_no_bn -mode train_eval

# moving the checkoints to the base_model_no_bn folder
mv ./checkpoints ./models/base_model_no_bn/

# training of deeper model
python src/main.py -model_dir ./models/deeper_model -mode train_eval

# moving the checkoints to the deeper folder
mv ./checkpoints ./models/deeper_model/

# predit test sample with different trained models 
python src/main.py -model_dir ./models/base_model \
 -weights ./models/base_model/checkpoints/model_7000.pth \
 -mode predict - pred_ix 0 1 2 3 4 5 6 7 8 9

python src/main.py -model_dir ./models/base_model_no_bn \
 -weights ./models/base_model_no_bn/checkpoints/model_7000.pth \
 -mode predict - pred_ix 0 1 2 3 4 5 6 7 8 9

python src/main.py -model_dir models/deeper_model \
 -weights ./models/deeper_model/checkpoints/model_7000.pth \
 -mode predict - pred_ix 0 1 2 3 4 5 6 7 8 9