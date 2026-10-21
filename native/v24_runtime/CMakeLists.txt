cmake_minimum_required(VERSION 3.16)
project(biped_v24_native_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# External dependencies are located through cache variables; nothing is vendored.
#   MUJOCO_ROOT  directory containing include/mujoco/mujoco.h and libmujoco.so*
#                (an unpacked MuJoCo release, or the pip package directory)
#   ORT_ROOT     unpacked ONNX Runtime release (include/, lib/); optional
# Without ORT_ROOT only the ONNX-independent parts and their tests are built.
# ---------------------------------------------------------------------------
set(MUJOCO_ROOT "" CACHE PATH "MuJoCo install directory (include/ and lib*/)")
set(ORT_ROOT "" CACHE PATH "ONNX Runtime install directory (include/ and lib/)")

if(NOT MUJOCO_ROOT)
  message(FATAL_ERROR "Set -DMUJOCO_ROOT=<dir with include/mujoco/mujoco.h and libmujoco.so*>")
endif()
find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h PATHS "${MUJOCO_ROOT}/include" NO_DEFAULT_PATH)
find_library(MUJOCO_LIB NAMES mujoco libmujoco.so.3 PATHS "${MUJOCO_ROOT}" "${MUJOCO_ROOT}/lib" NO_DEFAULT_PATH)
if(NOT MUJOCO_LIB)
  file(GLOB _mj_candidates "${MUJOCO_ROOT}/libmujoco.so.*" "${MUJOCO_ROOT}/lib/libmujoco.so.*")
  list(GET _mj_candidates 0 MUJOCO_LIB)
endif()
if(NOT MUJOCO_INCLUDE_DIR OR NOT MUJOCO_LIB)
  message(FATAL_ERROR "MuJoCo not found under MUJOCO_ROOT=${MUJOCO_ROOT}")
endif()
get_filename_component(MUJOCO_LIB_DIR "${MUJOCO_LIB}" DIRECTORY)
message(STATUS "MuJoCo: ${MUJOCO_LIB}")

find_package(OpenSSL REQUIRED)
find_package(PkgConfig QUIET)

set(CMAKE_BUILD_RPATH "${MUJOCO_LIB_DIR}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)

# ---------------------------------------------------------------------------
# Core library: observation assembly, event machine, contact/GRF, control law,
# runtime bindings, artifact hash check. No ONNX Runtime dependency.
# ---------------------------------------------------------------------------
add_library(biped_core STATIC
  src/json_lite.cpp
  src/runtime_bindings.cpp
  src/obs_builder.cpp
  src/contact_utils.cpp
  src/event_machine.cpp
  src/control_law.cpp
  src/manifest_verify.cpp
)
target_include_directories(biped_core PUBLIC include "${MUJOCO_INCLUDE_DIR}")
target_link_libraries(biped_core PUBLIC "${MUJOCO_LIB}" OpenSSL::Crypto)

enable_testing()
function(biped_test name)
  add_executable(${name} tests/${name}.cpp)
  target_include_directories(${name} PRIVATE tests)
  target_link_libraries(${name} PRIVATE biped_core)
  add_test(NAME ${name} COMMAND ${name} ${ARGN})
endfunction()

biped_test(test_runtime_bindings "${CMAKE_SOURCE_DIR}/bindings/runtime_bindings.template.json")
biped_test(test_obs_builder)
biped_test(test_control_law)
biped_test(test_event_machine)
biped_test(test_contact_grf)
biped_test(test_push_controller)
biped_test(test_camera_controller)
biped_test(test_startup_verification)

# ---------------------------------------------------------------------------
# SDL2 (gamepad) -- optional.
# ---------------------------------------------------------------------------
set(BIPED_HAVE_SDL2 OFF)
if(PkgConfig_FOUND)
  pkg_check_modules(SDL2 sdl2)
  if(SDL2_FOUND)
    set(BIPED_HAVE_SDL2 ON)
  endif()
endif()
if(BIPED_HAVE_SDL2)
  add_executable(test_gamepad tests/test_gamepad.cpp)
  target_include_directories(test_gamepad PRIVATE tests ${SDL2_INCLUDE_DIRS})
  target_link_libraries(test_gamepad PRIVATE biped_core ${SDL2_LIBRARIES} pthread)
  target_link_directories(test_gamepad PRIVATE ${SDL2_LIBRARY_DIRS})
  add_test(NAME test_gamepad COMMAND test_gamepad)
else()
  message(STATUS "SDL2 not found: gamepad test and runtime app are not built")
endif()

# ---------------------------------------------------------------------------
# ONNX Runtime wrapper, its contract test and the closed-loop app -- optional.
# ---------------------------------------------------------------------------
if(ORT_ROOT)
  find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h PATHS "${ORT_ROOT}/include" NO_DEFAULT_PATH)
  file(GLOB _ort_candidates "${ORT_ROOT}/lib/libonnxruntime.so*")
  list(GET _ort_candidates 0 ORT_LIB)
  if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIB)
    message(FATAL_ERROR "ONNX Runtime not found under ORT_ROOT=${ORT_ROOT}")
  endif()
  message(STATUS "ONNX Runtime: ${ORT_LIB}")
  set(CMAKE_BUILD_RPATH "${MUJOCO_LIB_DIR};${ORT_ROOT}/lib")

  add_library(biped_onnx STATIC src/onnx_actor.cpp)
  target_include_directories(biped_onnx PUBLIC include "${ORT_INCLUDE_DIR}")
  target_link_libraries(biped_onnx PUBLIC "${ORT_LIB}")

  add_executable(test_onnx_actor tests/test_onnx_actor.cpp)
  target_include_directories(test_onnx_actor PRIVATE tests)
  target_link_libraries(test_onnx_actor PRIVATE biped_onnx)

  # Generates a synthetic (random-weight) actor with the Python tool, then runs the
  # wrapper test on it. Skipped (not failed) if the Python deps are unavailable.
  set(BIPED_PYTHON "python3" CACHE STRING "Python with numpy+onnx and biped_locomotion importable")
  set(_syn_dir "${CMAKE_BINARY_DIR}/synthetic")
  add_test(NAME synthetic_actor_generation
           COMMAND ${BIPED_PYTHON} "${CMAKE_SOURCE_DIR}/../../tools/make_synthetic_actor.py" --output-dir "${_syn_dir}")
  add_test(NAME test_onnx_actor
           COMMAND test_onnx_actor "${_syn_dir}/actor.onnx" "${_syn_dir}/synthetic_io.bin")
  set_tests_properties(test_onnx_actor PROPERTIES DEPENDS synthetic_actor_generation)

  if(BIPED_HAVE_SDL2)
    add_executable(biped_v24_native apps/biped_v24_native.cpp)
    target_include_directories(biped_v24_native PRIVATE ${SDL2_INCLUDE_DIRS})
    target_link_libraries(biped_v24_native PRIVATE biped_core biped_onnx ${SDL2_LIBRARIES} pthread)
    target_link_directories(biped_v24_native PRIVATE ${SDL2_LIBRARY_DIRS})
    # Smoke tests of the closed-loop app on SYNTHETIC assets only: they verify that
    # bindings loading, model validation, settling, inference, the event machine and
    # telemetry run end to end. The random actor does not walk, so the app's own
    # locomotion pass criteria are expected to FAIL and are deliberately ignored here.
    add_executable(make_synthetic_assets tests/make_synthetic_assets.cpp)
    target_include_directories(make_synthetic_assets PRIVATE tests)
    target_link_libraries(make_synthetic_assets PRIVATE biped_core)
    add_test(NAME synthetic_runtime_assets COMMAND make_synthetic_assets "${_syn_dir}")
    set_tests_properties(synthetic_runtime_assets PROPERTIES DEPENDS synthetic_actor_generation)
    add_test(NAME smoke_runtime_synthetic
             COMMAND biped_v24_native --model "${_syn_dir}/synthetic_model.mjb"
                     --policy "${_syn_dir}/actor.onnx" --bindings "${_syn_dir}/synthetic_bindings.json"
                     --manifest "${_syn_dir}/EXPORT_MANIFEST.json" --out "${_syn_dir}/out" --duration 0.5)
    set_tests_properties(smoke_runtime_synthetic PROPERTIES DEPENDS synthetic_runtime_assets
                         PASS_REGULAR_EXPRESSION "done\\. ticks=25")
    add_test(NAME runtime_usage_error COMMAND biped_v24_native)
    set_tests_properties(runtime_usage_error PROPERTIES PASS_REGULAR_EXPRESSION "usage:")
    add_test(NAME runtime_rejects_unfilled_template
             COMMAND biped_v24_native --model "${_syn_dir}/synthetic_model.mjb"
                     --policy "${_syn_dir}/actor.onnx"
                     --bindings "${CMAKE_SOURCE_DIR}/bindings/runtime_bindings.template.json"
                     --out "${_syn_dir}/out" --duration 0.1)
    set_tests_properties(runtime_rejects_unfilled_template PROPERTIES DEPENDS synthetic_runtime_assets
                         PASS_REGULAR_EXPRESSION "EXCEPTION: runtime_bindings")
    # Optional on-screen viewer (src: include/biped_native/glfw_viewer.h).
    pkg_check_modules(GLFW glfw3)
    if(GLFW_FOUND)
      message(STATUS "GLFW found: building biped_v24_native with --render support")
      target_compile_definitions(biped_v24_native PRIVATE BIPED_HAVE_GLFW)
      target_include_directories(biped_v24_native PRIVATE ${GLFW_INCLUDE_DIRS})
      target_link_libraries(biped_v24_native PRIVATE ${GLFW_LIBRARIES})
      target_link_directories(biped_v24_native PRIVATE ${GLFW_LIBRARY_DIRS})
    endif()
  endif()
else()
  message(STATUS "ORT_ROOT not set: ONNX wrapper test and runtime app are not built")
endif()
