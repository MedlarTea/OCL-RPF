# Message defining a tracked person
#
Header      header          # Header containing timestamp etc. of this message
uint64      track_id        # unique identifier of the target, consistent over time
float32     target_confidence          # target confidence inferenced by the target classifier
bool        is_occluded     # if the track is currently not observable in a physical way
bool        is_matched      # if the track is currently matched by a detection
uint64      detection_id    # id of the corresponding detection in the current cycle (undefined if occluded)
duration    age             # age of the track

# The following fields are extracted from the Kalman state x and its covariance C

geometry_msgs/PoseWithCovariance       pose   # pose of the track (z value and orientation might not be set, check if corresponding variance on diagonal is > 99999)

geometry_msgs/TwistWithCovariance   twist     # velocity of the track (z value and rotational velocities might not be set, check if corresponding variance on diagonal is > 99999)

Box bounding_box
# float64[] feature
