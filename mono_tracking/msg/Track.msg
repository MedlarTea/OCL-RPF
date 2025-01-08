int32 id
# float64 age

geometry_msgs/Point pos
geometry_msgs/Vector3 vel
float32 trace
# float32[36] cov
float32[16] cov

Box box  # x1,y1,x2,y2
float32[2] last_associated
float32[2] expected_measurement_mean
float32[4] expected_measurement_cov
