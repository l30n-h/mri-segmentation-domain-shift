DATA?=$(shell dirname `pwd`)/data
DOCKER_FILE=Dockerfile
SRC?=$(shell dirname `pwd`)
TAG?=nnunet_container_ma
CODE_PATH=$(SRC)/code/nnUNet
NBPORT=8888

UNAME?=$(shell whoami)
UID=$(shell id -u)

## Set default target
.DEFAULT_GOAL := bash_rootless

.PHONY: build_rootless bash_rootless

## Build and run an image for rootless docker. The user in the container has root privileges to have the same privileges as the user running the image.
build_rootless: download_code
	docker build -f Dockerfile-rootless -t $(TAG)-rootless --build-arg USER=$(UNAME) --build-arg NBPORT=$(NBPORT) .

bash_rootless: build_rootless download_code
	docker run --gpus all --rm -v $(SRC)/code:/home/$(UNAME)/code -v $(DATA):/home/$(UNAME)/data -it --shm-size=10gb $(TAG)-rootless bash

notebook_rootless: build_rootless download_code
	docker run --gpus all --rm -v $(SRC)/code:/home/$(UNAME)/code -v $(DATA):/home/$(UNAME)/data -it -p $(NBPORT):$(NBPORT) -t $(TAG)-rootless

download_code:
	@if ! test -d $(CODE_PATH); then git clone https://github.com/MIC-DKFZ/nnUNet.git $(CODE_PATH); fi;