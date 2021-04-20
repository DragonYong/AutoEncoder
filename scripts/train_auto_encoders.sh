DATA=/media/turing/D741ADF8271B9526/DATA
OUTPUT=/media/turing/D741ADF8271B9526/OUTPUT
python AutoEncoders.py \
    --DATA=$DATA/tensorflow/keras/mnist.npz \
    --IMAGE=$OUTPUT/AE \
    --BATCH_SIZE=2 \
    --EPOCHS=3 \
    --LEARNING_RATE=1e-3
