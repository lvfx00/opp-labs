cmake_minimum_required(VERSION 3.10)
project(lab5 C)

set(CMAKE_C_STANDARD 11)

include_directories(/usr/include/mpi)

add_executable(lab5
        lab5.c)
