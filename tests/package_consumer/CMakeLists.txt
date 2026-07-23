cmake_minimum_required(VERSION 3.14)
project(mollm_package_consumer LANGUAGES CXX)

find_package(mollm CONFIG REQUIRED)

add_executable(mollm_package_consumer main.cpp)
target_link_libraries(mollm_package_consumer PRIVATE mollm::mollm)
