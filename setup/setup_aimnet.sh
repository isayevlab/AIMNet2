#!/bin/bash

# Arguments:
#   root (string): Directory where everything is stored (except Docker images).
#   -full (flag): If present, perform a full setup.

main() {
    root=$1
    full=false

    # Check for '-full' flag
    for arg in "$@"; do
        if [ "$arg" == "-full" ]; then
            full=true
            break
        fi
    done

    if [ "$full" == true ]; then
        mkdir -p "$root"
        cd "$root"

        # Download AIMNet2
        git clone https://github.com/hoelzerC/AIMNet2.git
        cd "$root/AIMNet2/setup"

        # Install Docker
        sudo bash install_docker.sh
    fi

    # Set up Docker image
    cd "$root/AIMNet2"
    docker build --platform linux/amd64 --pull --rm -f "docker/Dockerfile_cpu" -t aimnet-box "."
}

# Execute the script with provided arguments
main "$@"
