DATA=/media/turing/D741ADF8271B9526/DATA
OUTPUT=/media/turing/D741ADF8271B9526/OUTPUT
python VaritionalAutoEncoders.py \
    --DATA=$DATA/tensorflow/keras/fashion-mnist \
    --IMAGE=$OUTPUT/VAE \
    --BATCH_SIZE=2 \
    --EPOCHS=1 \
    --LEARNING_RATE=1e-3
