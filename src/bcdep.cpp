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



bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return ( i > j );
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



std::string type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }
    r += "C";
    r += (chans+'0');
    
    return r;
}

void backgroundSubtraction(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::ImageConstPtr& dimage, cv::Mat& origin_img) {

    // ROS_INFO("Image received from Kinect - Size: %dx%d",
    //          image->width, image->height);
    std::string ty =  type2str( origin_img.type() );
    printf("ORIGIN_IMAGE: %s %dx%d \n", ty.c_str(), origin_img.cols, origin_img.rows );
    cv::Mat in_rgb_img = cv_bridge::toCvShare(image, sensor_msgs::image_encodings::BGR8)->image;
    cv::Mat ori_rgb = in_rgb_img;
    cv::cvtColor(in_rgb_img, in_rgb_img, cv::COLOR_BGR2GRAY);
    std::string tdy =  type2str( in_rgb_img.type() );
    printf("NEW_IMAGE: %s %dx%d \n", tdy.c_str(), in_rgb_img.cols, in_rgb_img.rows );
    cv::GaussianBlur(in_rgb_img, in_rgb_img, cv::Size(3, 3), 0, 0);
    
    try {
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to '16uc1'.", image->encoding.c_str());
    }
    // // pMog->operator()(in_rgb_img, fgMask);

    // cv::Ptr<cv::BackgroundSubtractor> pBackSub;
    
    // pBackSub = cv::createBackgroundSubtractorMOG2();
    // // pBackSub = cv::createBackgroundSubtractorMOG2(10, 1.0, true);
    // // pBackSub = cv::createBackgroundSubtractorKNN();
    // cv::Mat fgMask;
    // // cv::Ptr<cv::BackgroundSubtractor> mog = cv::bgsegm::createBackgroundSubtractorMOG();
    // pBackSub->apply(in_rgb_img, fgMask);
    // // mog->apply(in_rgb_img, fgMask);
    cv::Mat dst;
    cv::absdiff(origin_img, in_rgb_img, dst);
    cv::threshold(dst,dst, 50, 255, cv::THRESH_BINARY);
    cv::inRange(dst, cv::Scalar(254), cv::Scalar(255), dst);

    cv::Mat in_depth_image = cv_bridge::toCvShare(dimage, sensor_msgs::image_encodings::TYPE_16UC1)->image;
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours( dst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
    cv::Point2f center;
    float radius;
    float f = 5.453;
    float u_rgb, v_rgb, depth_u, depth_v;
    if (contours.size() > 0) {
        std::sort(contours.begin(), contours.end(), compareContourAreas);
        cv::minEnclosingCircle(contours[0], center, radius);
        ROS_INFO("Radius: %.2f", radius);
        u_rgb = center.x;
        v_rgb = center.y;
        

        depth_u = 1.08 * u_rgb - 18.028;
        // depth_v = 1.08 * v_rgb - 29.743;
        depth_v = 1.0785 * v_rgb - 25.743;
        // depth_v = v_rgb;
        
        
        
        
        ///*
        if (depth_u < 0) depth_u = 0;
        if (depth_u > 640) depth_u = 640;
        depth_v = (depth_v<0) ? 0 : depth_v;
        depth_v = (depth_v>480) ? 480 : depth_v;
        
        // dz = static_cast<float>(in_depth_image.at<unsigned short>(int(depth_u), int(depth_v)));
        // //Note at this point dz can be 0!!!
        // dx = -(dz/f)*((depth_v - 320)*0.0093+0.063);
        // dy = -(dz/f)*((depth_u - 240)*0.0093+0.039);
        // printf("u: %.2f v: %.2f r: %.2f;    u: %.2f v: %.2f \n", u_rgb,v_rgb,radius, depth_u, depth_v);
        



        // visualize the detection 
        // if (radius < 20) radius = 20;
        
        double minVal, maxVal;
        cv::minMaxLoc(in_depth_image, &minVal, &maxVal);
        in_depth_image.convertTo(in_depth_image, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
        cv::circle(in_depth_image, cv::Point(int(depth_u), int(depth_v)), radius, cv::Scalar(0xffff), 1, cv::LINE_AA );
        cv::circle(dst, cv::Point(int(u_rgb), int(v_rgb)), int(radius), cv::Scalar(255,255,255), 1, cv::LINE_AA);
        // cv::circle(in_image, cv::Point(int(u_rgb), int(v_rgb)), 1, cv::Scalar(0,100,100), 1, cv::LINE_AA);
        // cv::circle(in_image, cv::Point(int(u_rgb), int(v_rgb)), int(radius), cv::Scalar(0,255,255), 1, cv::LINE_AA);
    }
    cv::imshow("viewRGB", ori_rgb);
    cv::imshow("viewDiffRGB", dst);
    cv::imshow("viewDepth", in_depth_image);
    cv::waitKey(30);



}


void backgroundSubtractionDepth(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::ImageConstPtr& dimage, cv::Mat& origin_img) {

    
    std::string ty =  type2str( origin_img.type() );
    printf("ORIGIN_IMAGE: %s %dx%d \n", ty.c_str(), origin_img.cols, origin_img.rows );
    cv::Mat in_depth_image = cv_bridge::toCvShare(dimage, sensor_msgs::image_encodings::TYPE_16UC1)->image;
    // in_depth_image.convertTo(in_depth_image, CV_8UC1);
    // cv::cvtColor(in_depth_image, in_depth_image, cv::COLOR_BGR2GRAY);
    std::string tdy =  type2str( in_depth_image.type() );
    printf("NEW_IMAGE: %s %dx%d \n", tdy.c_str(), in_depth_image.cols, in_depth_image.rows );
    // cv::Mat blurred;
    cv::GaussianBlur(in_depth_image, in_depth_image, cv::Size(3, 3), 0, 0);
    
    try {
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to '16uc1'.", dimage->encoding.c_str());
    }
    
    cv::Mat dst;
    // Do sth delete the dis == 255 and select the rest


    origin_img.convertTo(origin_img, CV_32FC1);
    cv::Mat n_depth32;
    in_depth_image.convertTo(n_depth32, CV_32FC1);
    cv::absdiff(origin_img, n_depth32, dst);
    // cv::threshold(dst,dst, 10, 100, cv::THRESH_BINARY);
    // cv::absdiff(origin_img, n_depth32, dst);
    cv::inRange(dst, cv::Scalar(1), cv::Scalar(200), dst);

    std::string type_dist =  type2str( dst.type() );
    printf("DIST_IMAGE: %s  \n", type_dist.c_str());
    // cv::absdiff(origin_img, in_depth_image, dst);cv::threshold(dst,dst, 100, 255, cv::THRESH_BINARY);
    // cv::inRange(dst, cv::Scalar(254), cv::Scalar(255), dst);
    // cv::threshold(dst, dst, max_binary_value)
    // cv::threshold(dst,dst, 100, 255, cv::THRESH_BINARY);

    cv::imshow("viewDepth", in_depth_image);
    cv::imshow("viewDiffDepth", dst);
    cv::waitKey(30);



}



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
    cv::namedWindow("viewRGB");
    cv::namedWindow("viewDiffDepth");
    cv::namedWindow("viewDepth");
    image_transport::ImageTransport it(nh);
    //  Code using RGB-D
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "camera/rgb/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "camera/depth/image_raw", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), rgb_sub, depth_sub);
    
    // cv::Mat origin_img = cv::imread("/home/yupeng/Image_Saver/middep.png", cv::IMREAD_GRAYSCALE);
    cv::Mat origin_img = cv::imread("/home/yupeng/Image_Saver/middep.png", -1);

    // cv::Mat origin_blue_img;
    cv::GaussianBlur(origin_img, origin_img, cv::Size(3, 3), 0, 0);
    sync.registerCallback(boost::bind(&backgroundSubtractionDepth, _1, _2, origin_img));
    


    
    ros::spin();
    cv::destroyWindow("viewRGB");
    cv::destroyWindow("viewDiffDepth");
    cv::destroyWindow("viewDepth");
}