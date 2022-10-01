# buiding:
# docker build -f docker/dockerfile -t deeplense .

docker run -it \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -v $PWD/logger:/workspace/DeepLense/logger \
    -v $PWD/data:/workspace/DeepLense/data \
    --gpus all \
    --privileged \
    deeplense:latest bash 
