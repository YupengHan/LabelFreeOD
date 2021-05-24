#include <ros/ros.h>
#include <image_transport/image_transport.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/bgsegm.hpp>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cuda.h>
#include <cuda_runtime.h>


void testmain(int size, int *c);
void find_loc_fine(float* pts, int ptnum, int* scores, float* xyz_limits, float* device_pred_xyz);




/*
    Helper Func to save syn Depth&RGB Image
    Used for caliberation
*/
void saveRGBDImage(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::ImageConstPtr& dimage) {
    ROS_INFO("Image received from Kinect - Size: %dx%d",
             msg->width, msg->height);
    
    cv::Mat in_image = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8)->image;
    cv::Mat in_depth_image = cv_bridge::toCvShare(dimage, sensor_msgs::image_encodings::TYPE_16UC1)->image;

    
    try {
        // cv::namedWindow("view");
        // cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
        
        cv::imshow("view", in_image);
        cv::imshow("viewDepth", in_depth_image);
        
        
        
        cv::imwrite("/home/yupeng/Image_Saver/midrgb.jpg", in_image);
        cv::imwrite("/home/yupeng/Image_Saver/middep.png", in_depth_image);
        cv::waitKey(30);
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}


/* Pixel Projection Caliberation
    RGB     (77,36) (528,50)
            (68,462) (535,450)
            (318, 236) (314,443)

    Depth   (69,9) (557,24)
            (56,471) (562,460)
            (318, 225) (321,452)

    
    (u,v)
    RGB->Depth: 

    u_d = 1.0821567469030684 * u_r - 18.028069050274368
    v_d = 1.0785184312136273 * v_r - 29.743432242898184
*/






int main (int argc, char **argv) {
    ros::init(argc, argv, "image_listener");
    ros::NodeHandle nh;
    // int num = 1;
	// int *p;
	// int s = num*sizeof(int);
	// p = (int *)malloc(s);
	// testmain(s,p);
    // printf("---------------------------------------------");
	// printf("%d\n",p[0]);
    // printf("---------------------------------------------");
    cv::namedWindow("view");
    cv::namedWindow("viewDepth");
    image_transport::ImageTransport it(nh);
    // Code using RGB-D
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "camera/rgb/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "camera/depth/image_raw", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), rgb_sub, depth_sub);
    

    sync.registerCallback(boost::bind(&saveRGBDImage, _1, _2));
    
    
    


    
    ros::spin();
    cv::destroyWindow("view");
    cv::destroyWindow("viewDepth");
}