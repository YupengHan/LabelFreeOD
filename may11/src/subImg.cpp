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


#include <iostream>
#include <fstream>

void testmain(int size, int *c);
void find_loc_fine(float* pts, int ptnum, int* scores, float* xyz_limits, float* device_pred_xyz);

std::vector<float> pred_pts;


void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    ROS_INFO("Image received from Kinect - Size: %dx%d",
             msg->width, msg->height);
    try {
        // cv::namedWindow("view");
        cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
        cv::waitKey(30);
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}





bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return ( i > j );
}



/*load xyz*/
void load_xyz(cv::Mat d_img, int u, int v, float detect_r, float* host_ob_xyz, float* xyz_limit, int* poinNum) {
    int r;
    if (detect_r < 10) {
        r = 10;
    }
    else if (detect_r < 20) {
        r = 20;
    }
    else if (detect_r < 40) {
        r = 40;
    }
    else {
        r = 80;
    }

    float f = 5.453;
    int i = 0;
    float min_x = 3000, min_y = 3000, min_z = 7000, max_x = -3000, max_y = -3000, max_z = 0;
    int delta = r/10;
    // std::cout << "delta: " << delta << std::endl;
    for (int cu = u - r; cu < u + r; cu += 1) {
        for (int cv = v - r; cv < v + r; cv += 1) {
            float cz = static_cast<float>(d_img.at<unsigned short>(int(cu), int(cv)));
            if (cz > 40) min_z = cz < min_z ? cz : min_z;
        }
    }
    // std::cout << "min_z: " << min_z << std::endl;


    for (int cu = u - r; cu < u + r; cu += delta) {
        for (int cv = v - r; cv < v + r; cv += delta) {
            float cz = static_cast<float>(d_img.at<unsigned short>(int(cu), int(cv)));
            if (cz != 0 && (cz-min_z) < 200) {
                float cx = -(cz/f)*((cv - 320)*0.0093+0.063);
                float cy = -(cz/f)*((cu - 240)*0.0093+0.039);
                host_ob_xyz[3*i + 0] = cx;
                host_ob_xyz[3*i + 1] = cy;
                host_ob_xyz[3*i + 2] = cz;
                min_x = cx < min_x ? cx : min_x;
                min_y = cy < min_y ? cy : min_y;
                min_z = cz < min_z ? cz : min_z;
                max_x = cx > max_x ? cx : max_x;
                max_y = cy > max_y ? cy : max_y;
                max_z = cz > max_z ? cz : max_z;
                i++;
            }
            // else {
            //     xyz[3*i + 0] = 0;
            //     xyz[3*i + 1] = 0;
            //     xyz[3*i + 2] = 0;
            // }
            // i++;
            
        }
    }
    poinNum[0] = i;
    xyz_limit[0] = min_x;
    xyz_limit[1] = max_x;
    xyz_limit[2] = min_y;
    xyz_limit[3] = max_y;
    xyz_limit[4] = min_z;
    xyz_limit[5] = max_z;
    // std::cout << xyz_limit[4] << " " << xyz_limit[5] << std::endl;
}


void savetxt(std::vector<float> pts) {
    std::ofstream outf("/home/yupeng/pred_pts.xyz");
    for (int l = 0; l < pts.size(); l+= 3) {
        outf << pts[l] << " " << pts[l+1] << " " << pts[l+2] << std::endl;
    }
    outf.close();
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





void bsLoc(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::ImageConstPtr& dimage,
    float* host_ob_xyz,float*  device_ob_xyz, 
    float* xyz_limit, float* device_xyz_limits,
    int* device_loc_scores,
    float* host_pred_xyz, float* device_pred_xyz) {
    // float* pred_pts
    // ROS_INFO("Image received from Kinect - Size: %dx%d",
    //          msg->width, msg->height);
    
    

    cv::Mat origin_img = cv::imread("/home/yupeng/Image_Saver/midrgb_new.jpg");
    cv::cvtColor(origin_img, origin_img, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(origin_img, origin_img, cv::Size(3, 3), 0, 0);
    
    
    
    cv::Mat in_image = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8)->image;
    cv::cvtColor(in_image, in_image, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(in_image, in_image, cv::Size(3, 3), 0, 0);
    cv::Mat dst;
    cv::absdiff(origin_img, in_image, dst);
    cv::threshold(dst,dst, 15, 255, cv::THRESH_BINARY);
    
    cv::inRange(dst, cv::Scalar(254), cv::Scalar(255), dst);
    

    cv::Mat in_depth_image = cv_bridge::toCvShare(dimage, sensor_msgs::image_encodings::TYPE_16UC1)->image;

    
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours( dst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
    cv::Point2f center;
    float radius;
    float f = 5.453;
    float u_rgb, v_rgb, depth_u, depth_v;
    int* poinNumPtr = (int*) malloc(sizeof(int));
    poinNumPtr[0] = 0;
    if (contours.size() > 0) {
        std::sort(contours.begin(), contours.end(), compareContourAreas);
        cv::minEnclosingCircle(contours[0], center, radius);
        u_rgb = center.x;
        v_rgb = center.y;
        

        depth_u = 1.082 * u_rgb - 19.028;
        depth_v = 1.0785 * v_rgb - 29.743;
        // depth_v = 1.0785 * v_rgb - 25.743;
        
        
        
        
        ///*
        if (depth_u < 0) depth_u = 0;
        if (depth_u > 640) depth_u = 640;
        depth_v = (depth_v<0) ? 0 : depth_v;
        depth_v = (depth_v>480) ? 480 : depth_v;
        
        printf("R: %.2f u: %.2f v: %.2f r: %.2f;    u: %.2f v: %.2f \n", radius, u_rgb,v_rgb,radius, depth_u, depth_v);

        load_xyz(in_depth_image, int(depth_v), int(depth_u), radius, host_ob_xyz, xyz_limit, poinNumPtr);
        // printf("u: %.2f v: %.2f r: %.2f; minX: %.2f; minY: %.2f; minZ: %.2f; poinNum: %d  \n", u_rgb,v_rgb,radius,xyz_limit[0], xyz_limit[2],xyz_limit[4], poinNumPtr[0]);
        int poinNum = poinNumPtr[0];
        cudaMemcpy(device_ob_xyz, host_ob_xyz, poinNum *3*sizeof(float), cudaMemcpyHostToDevice);
        int posenum = int(xyz_limit[1]-xyz_limit[0])*int(xyz_limit[3]-xyz_limit[2])*int(xyz_limit[5]-xyz_limit[4]);

        cudaMemcpy(device_xyz_limits, xyz_limit, 6*sizeof(float), cudaMemcpyHostToDevice);
        find_loc_fine(device_ob_xyz, poinNum, device_loc_scores, device_xyz_limits, device_pred_xyz);
        cudaMemcpy(host_pred_xyz, device_pred_xyz, 3*sizeof(float), cudaMemcpyDeviceToHost);
        printf("x: %.2f ;  y: %.2f ;   z: %.2f \n", host_pred_xyz[0], host_pred_xyz[1], host_pred_xyz[2]);
        // */
        

        //under some situation
        if (radius > 2 && host_pred_xyz[0] != -10000) {
            pred_pts.push_back(host_pred_xyz[0]);
            pred_pts.push_back(host_pred_xyz[1]);
            pred_pts.push_back(host_pred_xyz[2]);
        }
        if (pred_pts.size() > 30) {
            savetxt(pred_pts);
        } 
        



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
    
    try {
        // cv::namedWindow("view");
        // cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
        
        // cv::imshow("view", in_image);
        cv::imshow("viewDepth", in_depth_image);
        cv::imshow("view", dst);
        
        cv::waitKey(30);
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}


// void saveRGBDImage(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::ImageConstPtr& dimage) {
void saveRGBDImage(const sensor_msgs::Image& msg) {
    ROS_INFO("Image received from Kinect - Size: %dx%d",
             msg.width, msg.height);
    
    // sensor_msgs::ImageConstPtr rosimg = boost::shared_ptr<sensor_msgs::Image>(msg);
    
    // cv::Mat in_image = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8)->image;
    cv::Mat in_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
    // cv::Mat in_depth_image = cv_bridge::toCvShare(dimage, sensor_msgs::image_encodings::TYPE_16UC1)->image;

    
    try {
        
        cv::imshow("view", in_image);
        // cv::imshow("viewDepth", in_depth_image);
        
        
        
        cv::imwrite("/home/yupeng/Image_Saver/midrgb_new.jpg", in_image);
        // cv::imwrite("/home/yupeng/Image_Saver/middep.png", in_depth_image);
        cv::waitKey(30);
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg.encoding.c_str());
    }
    
    return;
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
    cv::namedWindow("view");
    cv::namedWindow("viewDepth");
    image_transport::ImageTransport it(nh);
    //  Code using RGB-D
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "camera/rgb/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "camera/depth/image_raw", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), rgb_sub, depth_sub);


    // bool saveImgOrNot = false;
    // std::cout << "calling the save img fn again argh " << std::endl;
    // if (!saveImgOrNot) {
    //     // sync.registerCallback(boost::bind(&saveRGBDImage, _1, _2));
    //     ros::Subscriber rgb_simple_sub = nh.subscribe("camera/rgb/image_raw", 1, saveRGBDImage);
    //     saveImgOrNot = true;
    // }

    // boost::shared_ptr<sensor_msgs::Image const> rgb_img_ptr;
    // sensor_msgs::Image rgb_img;
    // rgb_img_ptr = ros::topic::waitForMessage<sensor_msgs::Image>("camera/rgb/image_raw", saveRGBDImage);
    // if(rgb_img_ptr != NULL){
    //     rgb_img = *rgb_img_ptr;
    // }

    sensor_msgs::Image pc;
    
    pc  = *(ros::topic::waitForMessage<sensor_msgs::Image>("camera/rgb/image_raw", nh));
    saveRGBDImage(pc);

    std::cout << "came out of the saveimgornot bool thinge " << std::endl;
    
    

    size_t ob_xyz_size = 20*20*3*sizeof(float);
    float* host_ob_xyz = (float*) malloc(ob_xyz_size);
    float* device_ob_xyz;
    cudaMalloc((void **)&device_ob_xyz, ob_xyz_size);

    float* xyz_limit = (float*) malloc(6*sizeof(float));
    float* device_xyz_limits;
    cudaMalloc((void **)&device_xyz_limits, 6*sizeof(float));

    int* device_loc_scores;
    cudaMalloc((void **)&device_loc_scores, 100*100*400*sizeof(int)); //?

    float* host_pred_xyz = (float*) malloc(3*sizeof(float));
    float* device_pred_xyz;
    cudaMalloc((void **)&device_pred_xyz, 3*sizeof(float));
    
    while(ros::ok()){
        std::cout << " in the ros ok loop " << std::endl;
        
        


        
        // cv::Mat origin_img = cv::imread("/home/yupeng/Image_Saver/midrgb.jpg");
        // cv::cvtColor(origin_img, origin_img, cv::COLOR_BGR2GRAY);

        // // cv::Mat origin_blue_img;
        // cv::GaussianBlur(origin_img, origin_img, cv::Size(3, 3), 0, 0);
        
        
        // sync.registerCallback(boost::bind(&testcallback, _1, _2, 
        //     host_ob_xyz, device_ob_xyz, 
        //     xyz_limit, device_xyz_limits,
        //     device_loc_scores,
        //     host_pred_xyz, device_pred_xyz,
        //     origin_img));
        sync.registerCallback(boost::bind(&bsLoc, _1, _2, 
            host_ob_xyz, device_ob_xyz, 
            xyz_limit, device_xyz_limits,
            device_loc_scores,
            host_pred_xyz, device_pred_xyz));
        
        //Used for save RGB frame and Depth frame for calibration
        // sync.registerCallback(boost::bind(&saveRGBDImage, _1, _2));
        
        

        // Try to use background subtraction

        // cv::Mat origin_img = cv::imread("/home/yupeng/Image_Saver/middep.png", cv::IMREAD_GRAYSCALE);
        



        /*
        cv::Mat origin_img = cv::imread("/home/yupeng/Image_Saver/midrgb.jpg");
        cv::cvtColor(origin_img, origin_img, cv::COLOR_BGR2GRAY);

        // cv::Mat origin_blue_img;
        cv::GaussianBlur(origin_img, origin_img, cv::Size(3, 3), 0, 0);
        image_transport::Subscriber sub = it.subscribe("/camera/rgb/image_raw", 1, boost::bind(&backgroundSubtraction, _1, origin_img));
        */

        
        
        
        /*
        // cv::Mat origin_img = cv::imread("/home/yupeng/Image_Saver/middep.png", cv::IMREAD_GRAYSCALE);
        cv::Mat origin_img = cv::imread("/home/yupeng/Image_Saver/middep.png", -1);
        // origin_img.convertTo(origin_img, CV_8UC1);
        cv::GaussianBlur(origin_img, origin_img, cv::Size(3, 3), 0, 0);
        image_transport::Subscriber sub = it.subscribe("/camera/depth/image_raw", 1, boost::bind(&backgroundSubtractionDepth, _1, origin_img));
        */
        
        


        
        ros::spin();
    }

    std::cout << "came out of the loop " << std::endl;
    cv::destroyWindow("view");
    cv::destroyWindow("viewDepth");
}