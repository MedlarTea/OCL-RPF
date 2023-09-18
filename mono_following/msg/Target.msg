std_msgs/Header header

std_msgs/String state  # current following state
int64 target_id  # target id
float32[] classifier_confidences  # target confidence estimated by the target classifier
float64[] position  # target position
int64[] box # tl_x, tl_y, w, h

int64[] track_ids  # all track ids
float32[] confidences  # all confidences estimated by the target classifier
