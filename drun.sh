#!/bin/bash
drun() {
    sudo docker run --runtime=nvidia --gpus all opencb_docker "$@"
}

drun_no_cuda() {
    sudo docker run opencb_docker "$@"
}

helpmsg() {
    echo "Some simple usage instructions:"
    echo "1) drun --help                                ---     Display this message"
    echo "2) drun --no-cuda [arg1] [arg2] ... [argN]    ---     Run the container with the specified arguments and without Nvidia Cuda support"
    echo "3) drun [arg1] [arg2] ... [argN]              ---     Run the container with the specified arguments"
}

if [ "$1" == "--help" ]; then
    helpmsg
    exit 0
fi
if [ "$1" == "--no-cuda" ]; then
    shift
    drun_no_cuda "$@"
    exit 0
fi

drun "$@"