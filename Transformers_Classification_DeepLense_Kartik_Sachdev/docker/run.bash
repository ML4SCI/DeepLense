# buiding:
# docker build -f docker/dockerfile -t deeplense:latest .

docker run -it \
    -e DISPLAY=$DISPLAY \
    --net=host\
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --gpus all \
    --privileged \
    deeplense:latest bash 
