#include "biped_native/contact_utils.h"

#include <algorithm>

namespace biped {

bool GeomsInContact(const mjData* d, int nsize, int geom_a, int geom_b) {
  for (int i = 0; i < nsize; ++i) {
    const mjContact& c = d->contact[i];
    const bool pair_matches =
        (c.geom1 == geom_a && c.geom2 == geom_b) || (c.geom1 == geom_b && c.geom2 == geom_a);
    if (pair_matches && c.dist <= kContactDistThreshold) return true;
  }
  return false;
}

FootGrf ComputeFootGrf(const mjModel* m, const mjData* d, int floor_id, int left_foot_id,
                        int right_foot_id) {
  FootGrf out;
  for (int i = 0; i < d->ncon; ++i) {
    const mjContact& c = d->contact[i];
    if (c.dist > kContactDistThreshold) continue;

    const bool floor_is_geom1 = (c.geom1 == floor_id);
    const bool floor_is_geom2 = (c.geom2 == floor_id);
    if (!floor_is_geom1 && !floor_is_geom2) continue;
    const int foot_geom = floor_is_geom1 ? c.geom2 : c.geom1;
    const bool is_left = (foot_geom == left_foot_id);
    const bool is_right = (foot_geom == right_foot_id);
    if (!is_left && !is_right) continue;

    mjtNum result[6]{};
    mj_contactForce(m, d, i, result);
    const mjtNum* frame = c.frame;  // row-major 3x3: row0=normal, row1=tangent1, row2=tangent2
    double contact_force_world_z = 0.0;
    for (int row = 0; row < 3; ++row) {
      contact_force_world_z += result[row] * frame[row * 3 + 2];
    }
    // frame's normal (row0) points from geom1 to geom2 by MuJoCo convention.
    // Flip sign when the floor is geom2 so the result is always "force
    // applied by the floor to the foot," independent of geom id ordering.
    const double foot_force_world_z =
        floor_is_geom1 ? contact_force_world_z : -contact_force_world_z;
    const double fz = std::max(foot_force_world_z, 0.0);

    if (is_left) {
      out.left += fz;
    } else {
      out.right += fz;
    }
  }
  return out;
}

}  // namespace biped
