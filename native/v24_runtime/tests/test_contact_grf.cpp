// Sanity check for contact_utils.cpp's ComputeFootGrf on the synthetic test
// model: after settling, per-foot ground reaction force must be finite,
// non-negative, consistent with GeomsInContact, and sum to a sane fraction of
// body weight. Guards against the geom1/geom2 ordering sign bug recurring
// silently. Also reports the observed floor/foot geom orderings.
//
// Usage: test_contact_grf
#include <cmath>
#include <cstdio>
#include <exception>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include <mujoco/mujoco.h>

#include "biped_native/contact_utils.h"
#include "biped_native/obs_builder.h"
#include "biped_native/runtime_layout.h"
#include "synthetic_model.h"

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  mjModel* m = synthetic::MakeModel();
  const biped::RuntimeBindings bindings = synthetic::MakeBindings(m);
  mjData* d = mj_makeData(m);
  if (!d) {
    std::fprintf(stderr, "[test_contact_grf] mj_makeData failed\n");
    mj_deleteModel(m);
    return 1;
  }
  bool ok = true;

  try {
    if (m->nu != biped::kNumJoints) throw std::runtime_error("synthetic model must have 12 actuators");

    biped::ContactGeoms geoms = biped::ResolveContactGeoms(m, bindings);

    // Settle: initial pose, zero velocity, zero control, kSettleSteps of mj_step.
    for (int i = 0; i < m->nq; ++i) d->qpos[i] = bindings.initial_qpos[i];
    for (int i = 0; i < m->nv; ++i) d->qvel[i] = 0.0;
    mj_forward(m, d);
    for (int i = 0; i < biped::kNumJoints; ++i) d->ctrl[i] = 0.0;
    for (int s = 0; s < biped::kSettleSteps; ++s) mj_step(m, d);
    for (int i = 0; i < m->nv; ++i) d->qvel[i] = 0.0;
    mj_forward(m, d);

    const biped::FootGrf grf =
        biped::ComputeFootGrf(m, d, geoms.floor_id, geoms.left_foot_id, geoms.right_foot_id);
    const double total_grf = grf.left + grf.right;

    const bool left_contact = biped::GeomsInContact(d, d->ncon, geoms.floor_id, geoms.left_foot_id);
    const bool right_contact =
        biped::GeomsInContact(d, d->ncon, geoms.floor_id, geoms.right_foot_id);

    double total_model_mass = 0.0;
    for (int i = 0; i < m->nbody; ++i) total_model_mass += m->body_mass[i];
    const double gravity_magnitude = std::fabs(m->opt.gravity[2]);
    const double expected_weight = total_model_mass * gravity_magnitude;

    std::set<std::pair<int, int>> orderings;
    for (int i = 0; i < d->ncon; ++i) {
      const mjContact& c = d->contact[i];
      if (c.dist > biped::kContactDistThreshold) continue;
      const bool touches_left =
          (c.geom1 == geoms.left_foot_id || c.geom2 == geoms.left_foot_id) &&
          (c.geom1 == geoms.floor_id || c.geom2 == geoms.floor_id);
      const bool touches_right =
          (c.geom1 == geoms.right_foot_id || c.geom2 == geoms.right_foot_id) &&
          (c.geom1 == geoms.floor_id || c.geom2 == geoms.floor_id);
      if (touches_left || touches_right) orderings.insert({c.geom1, c.geom2});
    }

    const bool finite = std::isfinite(grf.left) && std::isfinite(grf.right);
    const bool nonnegative = grf.left >= 0.0 && grf.right >= 0.0;
    const bool nonzero_while_standing = total_grf > 1e-6;
    // Loose band: standing on two feet after settle should roughly support
    // body weight (not exactly, since settle may still have residual
    // acceleration/damping forces), but should not be near-zero or wildly
    // over/under actual weight.
    const bool reasonable_band =
        total_grf > 0.5 * expected_weight && total_grf < 1.5 * expected_weight;
    const bool left_grf_implies_contact = !(grf.left > 0.0) || left_contact;
    const bool right_grf_implies_contact = !(grf.right > 0.0) || right_contact;

    std::printf("[test_contact_grf] left_grf=%.4f N right_grf=%.4f N total=%.4f N\n", grf.left,
                grf.right, total_grf);
    std::printf("[test_contact_grf] left_contact=%s right_contact=%s (diagnostic only, not "
                "asserted)\n",
                left_contact ? "true" : "false", right_contact ? "true" : "false");
    std::printf("[test_contact_grf] total_model_mass=%.4f kg gravity=%.4f expected_weight=%.4f N\n",
                total_model_mass, gravity_magnitude, expected_weight);
    std::printf("[test_contact_grf] observed floor/foot geom orderings (geom1,geom2) with "
                "dist<=%.4f: ",
                biped::kContactDistThreshold);
    for (const auto& [g1, g2] : orderings) std::printf("(%d,%d) ", g1, g2);
    std::printf("\n");
    std::printf("[test_contact_grf] finite=%s nonnegative=%s nonzero_while_standing=%s "
                "reasonable_band[0.5,1.5]x=%s left_grf_implies_contact=%s "
                "right_grf_implies_contact=%s\n",
                finite ? "true" : "false", nonnegative ? "true" : "false",
                nonzero_while_standing ? "true" : "false", reasonable_band ? "true" : "false",
                left_grf_implies_contact ? "true" : "false",
                right_grf_implies_contact ? "true" : "false");

    ok = finite && nonnegative && nonzero_while_standing && reasonable_band &&
         left_grf_implies_contact && right_grf_implies_contact;

    mj_deleteData(d);
    mj_deleteModel(m);
    std::printf("[test_contact_grf] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[test_contact_grf] EXCEPTION: %s\n", e.what());
    mj_deleteData(d);
    mj_deleteModel(m);
    return 1;
  }
}
