cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(autoenctest)

set(CMAKE_CXX_STANDARD 17)


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(DOWNLOAD https://raw.githubusercontent.com/lvandeve/lodepng/refs/heads/master/lodepng.cpp ${CMAKE_SOURCE_DIR}/thirdparty/lodepng.cpp)
file(DOWNLOAD https://raw.githubusercontent.com/lvandeve/lodepng/refs/heads/master/lodepng.h ${CMAKE_SOURCE_DIR}/thirdparty/lodepng.h)

add_executable(autoenctest autoenctest.cpp ${CMAKE_SOURCE_DIR}/thirdparty/lodepng.cpp)
add_executable(autoenctest-tuned autoenctest-tuned.cpp ${CMAKE_SOURCE_DIR}/thirdparty/lodepng.cpp)

target_link_libraries(autoenctest-tuned "${TORCH_LIBRARIES}")
target_link_libraries(autoenctest "${TORCH_LIBRARIES}")

set_property(TARGET autoenctest PROPERTY CXX_STANDARD 17)
set_property(TARGET autoenctest-tuned PROPERTY CXX_STANDARD 17)


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET autoenctest
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:autoenctest>)
endif (MSVC)
