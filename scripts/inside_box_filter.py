import rospy

def get_inside_ratio(bbox_out, bbox_in):
    
    overlap_bbox=[0,0,0,0]
    overlap_bbox[0] = max(bbox_out[0], bbox_in[0]);
    overlap_bbox[1] = max(bbox_out[1], bbox_in[1]);
    overlap_bbox[2] = min(bbox_out[2], bbox_in[2]);
    overlap_bbox[3] = min(bbox_out[3], bbox_in[3]);	
    
    overlap_width = overlap_bbox[2] - overlap_bbox[0] + 1;
    overlap_height = overlap_bbox[3] - overlap_bbox[1] + 1;
    
    inside_ratio = 0.0;
    if (overlap_width>0 and overlap_height>0):
        overlap_area = overlap_width*overlap_height
        bbox_in_area = (bbox_in[2] - bbox_in[0] + 1) * (bbox_in[3] - bbox_in[1] + 1)
        inside_ratio = float(overlap_area)/float(bbox_in_area);
    
    return inside_ratio
    
def filter_inside_boxes(detections, inside_ratio_thresh = 0.8):
    #filter pedestrian bounding box inside mobilityaids bounding box
    
    for outside_det in detections:
        
        # check for mobility aids bboxes
        if outside_det['category_id'] > 1:
            for inside_det in detections:
                #check all pedestrian detections against mobility aids detection
                if inside_det['category_id'] is 1:
                    inside_ratio = get_inside_ratio(outside_det['bbox'], inside_det['bbox'])
                    if inside_ratio > inside_ratio_thresh:
                        rospy.logdebug("filtering pedestrian bbox inside bbox with class id %d, inside ratio: %f" % (outside_det['category_id'], inside_ratio))
                        detections.remove(inside_det)