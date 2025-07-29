FROM ubuntu:latest
LABEL authors="Claudiu"
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENTRYPOINT ["top", "-b"]