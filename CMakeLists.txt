cmake_minimum_required(VERSION 3.14)
project(isda-simm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)

# Plain C++ pricer (no AADC dependency)
add_executable(isda-simm
    model/cpp/main.cpp
    model/cpp/vanilla_irs.cpp
    model/cpp/equity_option.cpp
    model/cpp/inflation_swap.cpp
    model/cpp/fx_option.cpp
    model/cpp/xccy_swap.cpp
)
target_include_directories(isda-simm PRIVATE model/cpp)
target_compile_options(isda-simm PRIVATE -O3 -Wall -Wextra)
if(OpenMP_CXX_FOUND)
    target_link_libraries(isda-simm PRIVATE OpenMP::OpenMP_CXX)
endif()

# AADC SDK paths
set(AADC_ROOT "$ENV{HOME}/aadc_sdk" CACHE PATH "AADC SDK root directory")
set(AADC_INCLUDE "${AADC_ROOT}/include")
set(AADC_LIB_DIR "${AADC_ROOT}/lib")

# Check if AADC is available
if(EXISTS "${AADC_INCLUDE}/aadc/aadc.h")
    message(STATUS "AADC SDK found at ${AADC_ROOT}")

    # Common AADC compile/link settings
    function(add_aadc_target TARGET_NAME SOURCE_FILE)
        add_executable(${TARGET_NAME} ${SOURCE_FILE})
        target_include_directories(${TARGET_NAME} PRIVATE ${AADC_INCLUDE} model/cpp)
        target_compile_options(${TARGET_NAME} PRIVATE -O3 -mavx2 -mfma -Wall)
        target_link_directories(${TARGET_NAME} PRIVATE ${AADC_LIB_DIR})
        target_link_libraries(${TARGET_NAME} PRIVATE aadc-avx2)
        set_target_properties(${TARGET_NAME} PROPERTIES
            BUILD_RPATH "${AADC_LIB_DIR}"
            INSTALL_RPATH "${AADC_LIB_DIR}"
        )
    endfunction()

    # AADC pricers (one per instrument type)
    add_aadc_target(vanilla_irs_aadc model/cpp/vanilla_irs_aadc.cpp)
    add_aadc_target(equity_option_aadc model/cpp/equity_option_aadc.cpp)
    add_aadc_target(inflation_swap_aadc model/cpp/inflation_swap_aadc.cpp)
    add_aadc_target(fx_option_aadc model/cpp/fx_option_aadc.cpp)
    add_aadc_target(xccy_swap_aadc model/cpp/xccy_swap_aadc.cpp)

else()
    message(WARNING "AADC SDK not found at ${AADC_ROOT}. AADC targets will not be built.")
endif()
