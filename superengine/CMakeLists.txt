cmake_minimum_required(VERSION 3.24)
project(SuperEngine VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto")

# Optional build flags for CI
option(ENABLE_SANITIZER "Enable address/undefined sanitizers" OFF)
option(ENABLE_COVERAGE "Enable gcov code coverage" OFF)

if(ENABLE_SANITIZER)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

if(ENABLE_COVERAGE)
  add_compile_options(--coverage -O0)
  add_link_options(--coverage)
endif()

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

find_package(Threads REQUIRED)

add_library(engine
    engine/bitboard.cpp
    engine/movegen.cpp
    engine/position.cpp
    engine/nnue_eval.cpp
    engine/search.cpp
    engine/mcts.cpp)

target_include_directories(engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/engine)
target_link_libraries(engine PUBLIC Threads::Threads)

find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
  message(STATUS "Building Python bindings")
  pybind11_add_module(cpp_mcts
      python/py_mcts.cpp)
  target_link_libraries(cpp_mcts PRIVATE engine)
  if(TARGET onnxruntime)
    target_compile_definitions(cpp_mcts PRIVATE USE_ONNXRUNTIME)
    target_link_libraries(cpp_mcts PRIVATE onnxruntime)
  endif()
endif()

add_executable(test_movegen tests/test_movegen.cpp)
target_link_libraries(test_movegen PRIVATE engine Catch2::Catch2WithMain)
add_test(NAME test_movegen COMMAND test_movegen)

add_executable(test_sliding tests/test_sliding.cpp)
target_link_libraries(test_sliding PRIVATE engine Catch2::Catch2WithMain)
add_test(NAME test_sliding COMMAND test_sliding)

add_executable(test_position tests/test_position.cpp)
target_link_libraries(test_position PRIVATE engine Catch2::Catch2WithMain)
add_test(NAME test_position COMMAND test_position)

add_executable(test_perft tests/perft.cpp)
target_link_libraries(test_perft PRIVATE engine Catch2::Catch2WithMain)
add_test(NAME test_perft COMMAND test_perft)

add_executable(test_search tests/test_search.cpp)
target_link_libraries(test_search PRIVATE engine Catch2::Catch2WithMain)
add_test(NAME test_search COMMAND test_search)

add_executable(test_nnue tests/test_nnue.cpp)
target_link_libraries(test_nnue PRIVATE engine Catch2::Catch2WithMain)
add_test(NAME test_nnue COMMAND test_nnue)

enable_testing()
