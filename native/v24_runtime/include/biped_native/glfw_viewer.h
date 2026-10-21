// Visible on-screen GLFW window rendering a live mjvScene, driven by
// CameraController's mjvCamera state (camera_controller.h) -- the "additive,
// not a redesign" wiring that file's header comment anticipated once GLFW
// headers become available. This header is only compiled when GLFW was
// found (see CMakeLists.txt's `if(GLFW_FOUND)` guard); nothing else in this
// codebase depends on it, so its absence never blocks a build.
//
// UNVERIFIED AT WRITE TIME: libglfw3-dev was not yet installed in this
// environment when this file was written, so this has not been compiled or
// run. It follows MuJoCo's standard mjv/mjr render-loop pattern (the same
// one used by mujoco's own `simulate`/`basic` sample apps). Compile and run
// it (see biped_v24_native.cpp's --render flag) as the very first thing
// once libglfw3-dev is available, before trusting this description.
#pragma once

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <stdexcept>
#include <string>

namespace biped {

class GlfwViewer {
 public:
  explicit GlfwViewer(const mjModel* m) : m_(m) {
    if (!glfwInit()) throw std::runtime_error("glfw_viewer: glfwInit failed");
    glfw_initialized_ = true;

    window_ = glfwCreateWindow(1200, 900, "V24 Native Runtime", nullptr, nullptr);
    if (!window_) {
      glfwTerminate();
      throw std::runtime_error("glfw_viewer: glfwCreateWindow failed");
    }
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(0);  // this app owns its own 50Hz pacing; don't double-throttle on vsync

    mjv_defaultOption(&opt_);
    mjv_defaultScene(&scn_);
    mjv_makeScene(m_, &scn_, 2000);
    mjr_defaultContext(&con_);
    mjr_makeContext(m_, &con_, mjFONTSCALE_150);
    scene_ready_ = true;
  }

  ~GlfwViewer() {
    if (scene_ready_) {
      mjv_freeScene(&scn_);
      mjr_freeContext(&con_);
    }
    if (window_) glfwDestroyWindow(window_);
    if (glfw_initialized_) glfwTerminate();
  }

  GlfwViewer(const GlfwViewer&) = delete;
  GlfwViewer& operator=(const GlfwViewer&) = delete;

  // Renders one frame using the given camera state and the CURRENT contents
  // of `d` (caller must have already called mj_step/mj_forward). Returns
  // false if the window was closed (caller should stop rendering, but the
  // physics/control loop itself is unaffected -- this is display-only).
  bool RenderFrame(const mjData* d, const mjvCamera& cam) {
    if (glfwWindowShouldClose(window_)) return false;

    int width, height;
    glfwGetFramebufferSize(window_, &width, &height);
    mjrRect viewport = {0, 0, width, height};

    mjvCamera cam_copy = cam;  // mjv_updateScene takes a non-const mjvCamera*
    mjv_updateScene(m_, const_cast<mjData*>(d), &opt_, nullptr, &cam_copy, mjCAT_ALL, &scn_);
    mjr_render(viewport, &scn_, &con_);

    glfwSwapBuffers(window_);
    glfwPollEvents();
    return true;
  }

  bool ShouldClose() const { return glfwWindowShouldClose(window_); }

 private:
  const mjModel* m_;
  GLFWwindow* window_ = nullptr;
  bool glfw_initialized_ = false;
  bool scene_ready_ = false;
  mjvOption opt_{};
  mjvScene scn_{};
  mjrContext con_{};
};

}  // namespace biped
