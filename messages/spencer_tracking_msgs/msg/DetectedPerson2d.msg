# Message defining a 2d image bbox of a detected person
#

# Unique id of the detection, monotonically increasing over time
uint64          detection_id

# (Pseudo-)probabilistic value between 0.0 and 1.0 describing the detector's confidence in the detection
float64         confidence
int32       x               # central x of 2d image bbox
int32       y               # cemtral y of 2d image bbox
uint32      w               # width of 2d image bbox
uint32      h               # height of 2d image bbox
float32     depth           # distance from the camera in m

# float64[] feature
