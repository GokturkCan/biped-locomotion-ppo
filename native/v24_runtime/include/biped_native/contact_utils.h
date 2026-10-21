// Shared foot-contact and ground-reaction-force resolution, used by both
// obs_builder.cpp (contact flags in the observation) and event_machine.cpp
// (load fractions for the gait event machine), so there is exactly one
// definition of "foot touching the floor".
#pragma once

#include <mujoco/mujoco.h>

namespace biped {

// self._config.contact_dist_threshold.
inline constexpr double kContactDistThreshold = 0.001;

// True iff a floor/foot contact pair exists in d->contact[0:nsize) with
// dist <= kContactDistThreshold. Mirrors _foot_contacts' `active` mask
// -- NOT merely "a candidate pair
// exists in d->contact[]", which is gated by MuJoCo's (larger) collision
// margin instead.
bool GeomsInContact(const mjData* d, int nsize, int geom_a, int geom_b);

// Per-foot vertical (world-frame) ground reaction force, summed over all
// floor/foot contacts with dist <= kContactDistThreshold. Native port of
// _foot_vertical_ground_reaction_forces: for each
// active contact, mj_contactForce(m, d, i, result) gives [normal, tangent1,
// tangent2, torsion, roll1, roll2] in the contact frame -- the same
// normal/tangent1/tangent2 triple the JAX path assembles by hand from
// efc_force for a pyramidal friction cone. result[0:3] is transformed to
// world frame via d->contact[i].frame (row-major 3x3: row0=normal,
// row1=tangent1, row2=tangent2), exactly mirroring the JAX einsum, and the
// world-frame z component is clipped to >=0 and summed per foot -- same as
// `fz_each = max(force_world[:, 2], 0.0)`.
struct FootGrf {
  double left = 0.0;
  double right = 0.0;
};

FootGrf ComputeFootGrf(const mjModel* m, const mjData* d, int floor_id, int left_foot_id,
                        int right_foot_id);

}  // namespace biped
