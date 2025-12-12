# Root Makefile for chem277b project

.PHONY: all build run clean

all: build run

build:
	@$(MAKE) -C docker build

run:
	@$(MAKE) -C docker run

clean:
	@$(MAKE) -C docker clean
