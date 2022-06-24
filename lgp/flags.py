# Whether to use a taller grid that has space for observing microwaaves
TALL_GRID = True

# Whether to visualize intermediate maps in OpenCV windows during rollouts and training.
# Set True for development
GLOBAL_VIZ = False

# If False, a rollout is initialized by rotating left 4 times to observe the environment
# If True, the initialization uses a longer sequence that includes looking up and down
LONG_INIT = True


BIG_TRACE = False

# Compute pitch angle based on the angle between agent's camera and the closest voxel that
# is indicated in the subgoal argument mask.
HEURISTIC_PITCH = True